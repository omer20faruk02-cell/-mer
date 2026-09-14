"""
BIST Hisse Tarama ve Teknik Analiz Botu (Web Sitesi)
==================================================================

Bu script şunu yapar:
1. Aşağıdaki TICKERS listesindeki (veya varsa TICKERS_CSV dosyasındaki)
   tüm BİST hisselerini tarar.
2. Senin belirlediğin kriterlere göre filtreler:
   - Göreceli hacim (MIN_REL_VOLUME)
   - Hacim/ciro (MIN_VOLUME_TL)
   - Günlük değişim (MIN_CHANGE_PCT)
3. Filtreyi geçen HER hisseye teknik analiz uygular: RSI(14), MACD(12,26,9),
   basit destek/direnç (son N günün swing low/high'ı).
4. Hepsini 0-100 arası "teknik puan" ile sıralar.
5. En yüksek puanlı ilk TOP_N hisseyi (giriş fiyatı, stop-loss, take-profit
   önerileriyle) öne çıkararak "public/index.html" dosyasını üretir.

GitHub Actions ile her akşam piyasa kapandıktan sonra otomatik çalışır,
ürettiği sayfa GitHub Pages üzerinden yayınlanır. Senin tek yapman
gereken telefonunda o linki açmak.

ÖNEMLİ: Bu puanlar ve seviyeler kesin kazanç garantisi DEĞİLDİR. Senin
belirttiğin indikatörleri (RSI, MACD, destek/direnç) kurallara göre
birleştiren basit bir özet aracıdır; son kararı ve risk yönetimini
sen vermelisin. Yatırım tavsiyesi değildir.

Kurulum (bilgisayarda test etmek istersen):
    pip install yfinance pandas numpy --break-system-packages
    python bist_tarama_bot.py
"""

import csv
import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# ------------------------------------------------------------------
# AYARLAR — istediğin gibi değiştir
# ------------------------------------------------------------------

# Opsiyonel: TradingView'den export ettiğin bir CSV kullanmak istersen
# dosya adını buraya yaz ve repo'ya ekle. Yoksa aşağıdaki TICKERS
# listesi (statik BİST evreni) kullanılır — otomasyon için bu yeterli,
# her gün elle güncellemene gerek yok; script zaten kriterleri her gün
# yeniden kendisi hesaplıyor.
TICKERS_CSV = "tv_screener.csv"

TICKERS = [
    "THYAO", "ASELS", "KCHOL", "SAHOL", "GARAN", "AKBNK", "ISCTR", "YKBNK",
    "TUPRS", "BIMAS", "EREGL", "SISE", "PGSUS", "FROTO", "TOASO", "TCELL",
    "KOZAL", "KOZAA", "SASA", "ASTOR", "ENJSA", "ODAS", "OYAKC", "TAVHL",
    "ARCLK", "VESTL", "PETKM", "HEKTS", "ALARK", "GUBRF", "KRDMD", "EGEEN",
    "SMRTG", "MPARK", "TTKOM", "ULKER", "VAKBN", "HALKB", "DOAS", "CIMSA",
    "AKSA", "AEFES", "BRSAN", "CCOLA", "ENKAI", "KONTR", "MGROS", "SOKM",
    "TSKB", "YATAS", "TKFEN", "OTKAR", "TTRAK", "KARSN", "BRISA", "GOODY",
    "IZMDC", "KMPUR", "SELEC", "CWENE", "ZOREN", "AGHOL", "BERA", "BIOEN",
    "EUPWR", "GESAN", "IZENR", "MIATK", "PASEU", "REEDR", "TABGD", "VESBE",
    "AKSEN", "ANHYT", "ANSGR", "BASGZ", "BFREN", "BTCIM", "ECILC", "ECZYT",
    "FENER", "GWIND", "IPEKE", "ISDMR", "KLSER", "KONYA", "LMKDC", "MAVI",
    "NTGAZ", "OYAYO", "PENTA", "QUAGR", "SOKE",  "TMSN",  "TURSG", "VAKKO",
]

MIN_REL_VOLUME = 3.0        # göreceli hacim eşiği (kendi stratejine göre değiştir)
MIN_VOLUME_TL = 50_000_000  # TL cirosu eşiği
MIN_CHANGE_PCT = 5.0        # günlük değişim eşiği (%)
REL_VOLUME_LOOKBACK = 20    # ortalama hacim için kaç günlük geçmiş kullanılsın
HISTORY_PERIOD = "6mo"      # indikatörler için ne kadar geçmiş veri çekilsin

TOP_N = 3                   # sitede kaç hisse öne çıksın
SL_BUFFER_PCT = 1.5         # stop-loss'u destek seviyesinin yüzde kaçı altına koysun
TP_BUFFER_PCT = 1.0         # take-profit'i direnç seviyesinin yüzde kaçı altına koysun


# ------------------------------------------------------------------
# TICKER LİSTESİNİ YÜKLEME (CSV varsa öncelik, yoksa statik liste)
# ------------------------------------------------------------------

def load_tickers() -> list[str]:
    if os.path.exists(TICKERS_CSV):
        tickers = []
        with open(TICKERS_CSV, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            symbol_col = None
            for candidate in ("Ticker", "Symbol", "Sembol", "Hisse"):
                if reader.fieldnames and candidate in reader.fieldnames:
                    symbol_col = candidate
                    break
            if symbol_col is None and reader.fieldnames:
                symbol_col = reader.fieldnames[0]
            for row in reader:
                raw = row.get(symbol_col, "").strip()
                if not raw:
                    continue
                ticker = raw.split(":")[-1].strip().upper()
                if ticker:
                    tickers.append(ticker)
        if tickers:
            print(f"{len(tickers)} hisse '{TICKERS_CSV}' dosyasından okundu.")
            return tickers
    print(f"'{TICKERS_CSV}' bulunamadı, statik liste kullanılıyor "
          f"({len(TICKERS)} hisse).")
    return TICKERS


# ------------------------------------------------------------------
# VERİ ÇEKME
# ------------------------------------------------------------------

def fetch_data(ticker: str):
    symbol = f"{ticker}.IS"
    try:
        df = yf.download(symbol, period=HISTORY_PERIOD, interval="1d",
                          progress=False, auto_adjust=True)
        if df is None or df.empty or len(df) < REL_VOLUME_LOOKBACK + 5:
            return None
        return df.dropna()
    except Exception as e:
        print(f"  [uyarı] {ticker}: veri çekilemedi ({e})")
        return None


# ------------------------------------------------------------------
# İNDİKATÖRLER
# ------------------------------------------------------------------

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def find_support_resistance(df: pd.DataFrame, window: int = 20):
    recent = df.tail(window)
    support = float(recent["Low"].min())
    resistance = float(recent["High"].max())
    return support, resistance


# ------------------------------------------------------------------
# TARAMA (SCREENING)
# ------------------------------------------------------------------

def screen_stock(ticker: str, df: pd.DataFrame):
    last_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2])
    change_pct = (last_close - prev_close) / prev_close * 100

    last_volume = float(df["Volume"].iloc[-1])
    avg_volume = float(df["Volume"].iloc[-(REL_VOLUME_LOOKBACK + 1):-1].mean())
    rel_volume = last_volume / avg_volume if avg_volume > 0 else 0

    volume_tl = last_volume * last_close

    passes = (
        rel_volume > MIN_REL_VOLUME
        and volume_tl > MIN_VOLUME_TL
        and change_pct > MIN_CHANGE_PCT
    )
    if not passes:
        return None

    return {
        "ticker": ticker,
        "close": last_close,
        "change_pct": change_pct,
        "rel_volume": rel_volume,
        "volume_tl": volume_tl,
    }


# ------------------------------------------------------------------
# TEKNİK ANALİZ PUANI + TRADE SEVİYELERİ
# ------------------------------------------------------------------

def analyze_stock(df: pd.DataFrame) -> dict:
    close = df["Close"]
    rsi = compute_rsi(close).iloc[-1]
    macd_line, signal_line, hist = compute_macd(close)
    macd_last, signal_last, hist_last = (
        macd_line.iloc[-1], signal_line.iloc[-1], hist.iloc[-1]
    )
    hist_prev = hist.iloc[-2]
    support, resistance = find_support_resistance(df)
    last_close = float(close.iloc[-1])

    score = 50
    notes = []

    if rsi < 30:
        score += 15
        notes.append(f"RSI {rsi:.1f} — aşırı satım, tepki alım ihtimali")
    elif rsi < 50:
        score += 8
        notes.append(f"RSI {rsi:.1f} — nötr/hafif güçlü")
    elif rsi < 70:
        score += 2
        notes.append(f"RSI {rsi:.1f} — güçlü ama aşırı alım değil")
    else:
        score -= 15
        notes.append(f"RSI {rsi:.1f} — aşırı alım, geri çekilme riski")

    if macd_last > signal_last and hist_last > hist_prev:
        score += 15
        notes.append("MACD sinyal üstünde ve momentum artıyor (pozitif)")
    elif macd_last > signal_last:
        score += 7
        notes.append("MACD sinyal üstünde ama momentum zayıflıyor")
    elif macd_last < signal_last and hist_last > hist_prev:
        score += 3
        notes.append("MACD sinyal altında ama toparlanma başlıyor")
    else:
        score -= 10
        notes.append("MACD sinyal altında, momentum negatif")

    range_span = resistance - support if resistance > support else 1
    pos_in_range = (last_close - support) / range_span
    if pos_in_range > 0.9:
        score -= 12
        notes.append("Fiyat direnç bölgesine çok yakın")
    elif pos_in_range < 0.3:
        score += 10
        notes.append("Fiyat desteğe yakın, yukarı yer var")
    else:
        score += 3
        notes.append("Fiyat destek-direnç aralığının ortasında")

    score = max(0, min(100, score))

    stop_loss = support * (1 - SL_BUFFER_PCT / 100)
    take_profit = resistance * (1 - TP_BUFFER_PCT / 100)
    risk = last_close - stop_loss
    reward = take_profit - last_close
    risk_reward = reward / risk if risk > 0 else float("nan")

    return {
        "score": score,
        "rsi": rsi,
        "support": support,
        "resistance": resistance,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_reward": risk_reward,
        "notes": notes,
    }


# ------------------------------------------------------------------
# WEB SİTESİ (GitHub Pages için statik HTML)
# ------------------------------------------------------------------

def build_html_report(results: list[dict]) -> str:
    now_str = datetime.now().strftime("%d.%m.%Y %H:%M")

    def card_html(r, highlight=False):
        cls = "card top" if highlight else "card"
        rr = f"1:{r['risk_reward']:.2f}" if not np.isnan(r["risk_reward"]) else "-"
        notes_html = "".join(f"<li>{n}</li>" for n in r["notes"])
        return f"""
        <div class="{cls}">
          <div class="card-head">
            <span class="ticker">{r['ticker']}</span>
            <span class="score">{r['score']}/100</span>
          </div>
          <div class="row"><span>Kapanış</span><b>{r['close']:.2f} TL</b></div>
          <div class="row"><span>Değişim</span><b>%{r['change_pct']:.2f}</b></div>
          <div class="row"><span>Göreceli Hacim</span><b>{r['rel_volume']:.2f}x</b></div>
          <div class="row"><span>Stop-Loss</span><b class="sl">{r['stop_loss']:.2f} TL</b></div>
          <div class="row"><span>Take-Profit</span><b class="tp">{r['take_profit']:.2f} TL</b></div>
          <div class="row"><span>Risk/Ödül</span><b>{rr}</b></div>
          <ul class="notes">{notes_html}</ul>
        </div>"""

    top_cards = "".join(card_html(r, highlight=True) for r in results[:TOP_N])
    other_rows = "".join(
        f"<tr><td>{r['ticker']}</td><td>{r['score']}</td>"
        f"<td>%{r['change_pct']:.2f}</td><td>{r['rel_volume']:.2f}x</td>"
        f"<td>{r['volume_tl']:,.0f} TL</td></tr>"
        for r in results[TOP_N:]
    )

    empty_msg = "" if results else "<p class='empty'>Bugün kriterlere uyan hisse bulunamadı.</p>"

    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BİST Tarama Botu</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#0f1115; color:#e8e8ea; margin:0; padding:16px; }}
  h1 {{ font-size:1.3rem; margin-bottom:2px; }}
  .updated {{ color:#9a9aa5; font-size:0.85rem; margin-bottom:18px; }}
  .criteria {{ color:#9a9aa5; font-size:0.8rem; margin-bottom:20px; }}
  .cards {{ display:flex; flex-direction:column; gap:14px; margin-bottom:24px; }}
  .card {{ background:#1b1e26; border-radius:12px; padding:14px 16px; border:1px solid #2a2e38; }}
  .card.top {{ border-color:#3d8bfd; }}
  .card-head {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }}
  .ticker {{ font-size:1.15rem; font-weight:700; }}
  .score {{ background:#3d8bfd22; color:#3d8bfd; padding:3px 10px; border-radius:20px; font-weight:600; font-size:0.9rem; }}
  .row {{ display:flex; justify-content:space-between; font-size:0.9rem; padding:3px 0; color:#c8c8d0; }}
  .sl {{ color:#ff6b6b; }}
  .tp {{ color:#51cf66; }}
  .notes {{ margin:8px 0 0; padding-left:18px; font-size:0.8rem; color:#9a9aa5; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
  th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid #2a2e38; }}
  th {{ color:#9a9aa5; font-weight:500; }}
  .disclaimer {{ margin-top:24px; font-size:0.78rem; color:#7a7a85; line-height:1.4; }}
  .empty {{ color:#9a9aa5; }}
  h2 {{ font-size:1rem; color:#c8c8d0; margin-top:0; }}
</style>
</head>
<body>
  <h1>📊 BİST Tarama Botu</h1>
  <div class="updated">Son güncelleme: {now_str}</div>
  <div class="criteria">Kriter: göreceli hacim &gt;{MIN_REL_VOLUME}, hacim &gt;{MIN_VOLUME_TL:,.0f} TL,
    değişim &gt;%{MIN_CHANGE_PCT} — {len(results)} hisse eşleşti</div>

  {empty_msg}

  <h2>🏆 En yüksek puanlı ilk {TOP_N} hisse</h2>
  <div class="cards">{top_cards}</div>

  {"<h2>Diğer eşleşenler</h2><table><tr><th>Hisse</th><th>Puan</th><th>Değişim</th><th>Gör. Hacim</th><th>Ciro</th></tr>" + other_rows + "</table>" if other_rows else ""}

  <div class="disclaimer">⚠️ Bu sayfadaki puanlar, stop-loss ve take-profit seviyeleri
  kesin kazanç garantisi değildir. RSI/MACD/destek-direnç'i kurallara göre
  birleştiren otomatik bir özet aracıdır, yatırım tavsiyesi değildir.
  Son kararı ve risk yönetimini sen almalısın.</div>
</body>
</html>"""


# ------------------------------------------------------------------
# ANA AKIŞ
# ------------------------------------------------------------------

def main():
    print(f"BIST Tarama — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    tickers = load_tickers()

    results = []
    for ticker in tickers:
        df = fetch_data(ticker)
        if df is None:
            continue
        screened = screen_stock(ticker, df)
        if screened is None:
            continue
        analysis = analyze_stock(df)
        screened.update(analysis)
        results.append(screened)

    results.sort(key=lambda r: r["score"], reverse=True)

    print(f"{len(results)} hisse kriterlere uydu.")

    # Web sitesi (GitHub Pages) için HTML üret
    os.makedirs("public", exist_ok=True)
    html = build_html_report(results)
    with open("public/index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("public/index.html oluşturuldu.")


if __name__ == "__main__":
    main()
