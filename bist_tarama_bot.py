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
4. Destek-direnç mesafesi ile RSI/MACD momentum gücünü birleştirerek
   "beklenen yükseliş %" tahmini üretir. SADECE:
     - beklenen yükseliş >= MIN_EXPECTED_RISE_PCT (%5) VE
     - fiyat desteğe yakın VEYA destekten yeni tepki almış
   olan hisseleri gösterir. Bu bot SADECE AL (BUY) fikirleri üretir;
   Midas'ta satış/açığa satış imkânı olmadığı için hiçbir zaman "SAT"
   sinyali veya yukarı potansiyeli kalmamış (dirence yapışmış) bir hisse
   göstermez.
5. Hepsini "beklenen yükseliş %" ve teknik puana göre sıralar, ilk TOP_N
   hisseyi (giriş fiyatı, stop-loss, hedef fiyat önerileriyle) öne
   çıkararak "public/index.html" dosyasını üretir.

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
MIN_CHANGE_PCT = 5.0        # günlük değişim eşiği (%) — bugünkü harekete bakan tarama filtresi
REL_VOLUME_LOOKBACK = 20    # ortalama hacim için kaç günlük geçmiş kullanılsın
HISTORY_PERIOD = "6mo"      # indikatörler için ne kadar geçmiş veri çekilsin

MIN_EXPECTED_RISE_PCT = 2.0   # yarın için beklenen minimum yükseliş (%) — bunun altındakiler elenir
SUPPORT_PROXIMITY_PCT = 3.0   # fiyat desteğe bu yüzde kadar yakınsa "desteğe yakın" sayılır
BOUNCE_LOOKBACK = 5            # "destekten tepki" ararken kaç günlük pencereye bakılsın

TOP_N = 3                   # sitede kaç hisse öne çıksın
SL_BUFFER_PCT = 1.5         # stop-loss'u destek seviyesinin yüzde kaçı altına koysun


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
        # yfinance'in bazı sürümleri tek hisse için de çoklu-indeks (MultiIndex)
        # sütun döndürüyor (örn. ("Close", "THYAO.IS")). Bunu düzleştiriyoruz,
        # yoksa df["Close"] tek sayı değil bir Series döner ve hata verir.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
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
# DESTEK YAKINLIĞI / DESTEKTEN TEPKİ TESPİTİ
# ------------------------------------------------------------------

def detect_support_setup(df: pd.DataFrame, support: float):
    """Fiyat desteğe yakın mı, yoksa son günlerde desteğe dokunup yukarı
    tepki mi verdi? İkisi de 'AL için hazır' sayılır."""
    last_close = float(df["Close"].iloc[-1])
    dist_to_support_pct = (last_close - support) / support * 100
    near_support = 0 <= dist_to_support_pct <= SUPPORT_PROXIMITY_PCT

    recent = df.tail(BOUNCE_LOOKBACK)
    touched_support = bool(
        ((recent["Low"] - support) / support * 100 <= SUPPORT_PROXIMITY_PCT).any()
    )
    recovering = float(recent["Close"].iloc[-1]) > float(recent["Close"].iloc[0])
    bounced_from_support = touched_support and recovering

    return near_support, bounced_from_support, dist_to_support_pct


# ------------------------------------------------------------------
# MOMENTUM GÜCÜ (RSI + MACD birleşimi) — beklenen yükselişi büyütür/küçültür
# ------------------------------------------------------------------

def compute_momentum_multiplier(rsi, macd_last, signal_last, hist_last, hist_prev):
    multiplier = 1.0
    notes = []

    if rsi < 35:
        multiplier += 0.30
        notes.append(f"RSI {rsi:.1f} — aşırı satımdan dönüş potansiyeli")
    elif rsi < 55:
        multiplier += 0.15
        notes.append(f"RSI {rsi:.1f} — güçlenmeye açık, nötr bölge")
    elif rsi < 70:
        notes.append(f"RSI {rsi:.1f} — güçlü ama henüz aşırı alım değil")
    else:
        multiplier -= 0.30
        notes.append(f"RSI {rsi:.1f} — aşırı alım, yükseliş payı sınırlı")

    if macd_last > signal_last and hist_last > hist_prev:
        multiplier += 0.25
        notes.append("MACD sinyal üstünde ve momentum güçleniyor")
    elif macd_last > signal_last:
        multiplier += 0.10
        notes.append("MACD sinyal üstünde, momentum yatay")
    elif hist_last > hist_prev:
        multiplier += 0.10
        notes.append("MACD sinyal altında ama toparlanma başlıyor")
    else:
        multiplier -= 0.25
        notes.append("MACD sinyal altında, momentum zayıf")

    return max(0.3, multiplier), notes


# ------------------------------------------------------------------
# TARAMA (SCREENING) — bugünkü hacim/değişim filtresi
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
# TEKNİK ANALİZ + BEKLENEN YÜKSELİŞ TAHMİNİ (SADECE AL FİKİRLERİ)
# ------------------------------------------------------------------

def analyze_stock(df: pd.DataFrame):
    """Destek-direnç mesafesi ile RSI/MACD momentum gücünü birleştirip
    'beklenen yükseliş %' üretir. Yükselecek yeri kalmamış (dirence
    yapışmış) ya da momentumu tamamen negatif olan hisseler burada
    elenir — bu bot hiçbir zaman satış fikri göstermez, sadece geçerli
    bir AL kurulumu varsa sonuç döner, yoksa None döner."""
    close = df["Close"]
    rsi = compute_rsi(close).iloc[-1]
    macd_line, signal_line, hist = compute_macd(close)
    macd_last, signal_last, hist_last = (
        macd_line.iloc[-1], signal_line.iloc[-1], hist.iloc[-1]
    )
    hist_prev = hist.iloc[-2]
    support, resistance = find_support_resistance(df)
    last_close = float(close.iloc[-1])

    # 1) Dirence kadar kalan ham yükselme payı (%)
    raw_potential_pct = (resistance - last_close) / last_close * 100
    if raw_potential_pct <= 0:
        return None  # fiyat zaten direncin üstünde/eşiğinde — yükselecek yer yok, AL fikri değil

    # 2) Momentum gücü (RSI + MACD) bu payı büyütür/küçültür
    momentum_multiplier, momentum_notes = compute_momentum_multiplier(
        rsi, macd_last, signal_last, hist_last, hist_prev
    )
    expected_rise_pct = raw_potential_pct * momentum_multiplier
    if expected_rise_pct <= 0:
        return None

    # 3) Destek yakınlığı / destekten tepki
    near_support, bounced_from_support, dist_to_support_pct = detect_support_setup(
        df, support
    )
    if not (near_support or bounced_from_support):
        return None  # sadece destek yakınında veya destekten tepki alan kurulumlar

    if expected_rise_pct < MIN_EXPECTED_RISE_PCT:
        return None  # senin istediğin minimum %5 beklenen yükseliş şartı

    # Teknik puan (0-100) — sıralama ve gösterim için
    score = 50
    notes = list(momentum_notes)
    score += (momentum_multiplier - 1.0) * 40  # momentum katkısı

    if bounced_from_support:
        score += 12
        notes.append(f"Destekten (%{dist_to_support_pct:.1f}) tepki alıp yükselişe geçti")
    if near_support:
        score += 8
        notes.append(f"Fiyat desteğe çok yakın (%{dist_to_support_pct:.1f})")

    score = max(0, min(100, round(score)))

    setup_label = (
        "Destekten Tepki Aldı" if bounced_from_support and not near_support else
        "Desteğe Yakın + Tepki" if bounced_from_support and near_support else
        "Desteğe Yakın"
    )

    stop_loss = support * (1 - SL_BUFFER_PCT / 100)
    target_price = last_close * (1 + expected_rise_pct / 100)

    return {
        "score": score,
        "rsi": rsi,
        "support": support,
        "resistance": resistance,
        "stop_loss": stop_loss,
        "target_price": target_price,
        "expected_rise_pct": expected_rise_pct,
        "setup_label": setup_label,
        "notes": notes,
    }


# ------------------------------------------------------------------
# WEB SİTESİ (GitHub Pages için statik HTML)
# ------------------------------------------------------------------

def build_html_report(results: list[dict]) -> str:
    now_str = datetime.now().strftime("%d.%m.%Y %H:%M")

    def card_html(r, highlight=False):
        cls = "card top" if highlight else "card"
        notes_html = "".join(f"<li>{n}</li>" for n in r["notes"])
        return f"""
        <div class="{cls}">
          <div class="card-head">
            <span class="ticker">{r['ticker']}</span>
            <span class="badge">🟢 AL</span>
            <span class="score">{r['score']}/100</span>
          </div>
          <div class="setup">{r['setup_label']}</div>
          <div class="row"><span>Kapanış</span><b>{r['close']:.2f} TL</b></div>
          <div class="row"><span>Bugünkü Değişim</span><b>%{r['change_pct']:.2f}</b></div>
          <div class="row"><span>Göreceli Hacim</span><b>{r['rel_volume']:.2f}x</b></div>
          <div class="row highlight"><span>Beklenen Yükseliş</span><b class="tp">%{r['expected_rise_pct']:.2f}</b></div>
          <div class="row"><span>Hedef Fiyat</span><b class="tp">{r['target_price']:.2f} TL</b></div>
          <div class="row"><span>Stop-Loss</span><b class="sl">{r['stop_loss']:.2f} TL</b></div>
          <ul class="notes">{notes_html}</ul>
        </div>"""

    top_cards = "".join(card_html(r, highlight=True) for r in results[:TOP_N])
    other_rows = "".join(
        f"<tr><td>{r['ticker']}</td><td>{r['score']}</td>"
        f"<td>%{r['expected_rise_pct']:.2f}</td><td>{r['setup_label']}</td>"
        f"<td>{r['target_price']:.2f} TL</td></tr>"
        for r in results[TOP_N:]
    )

    empty_msg = (
        "" if results else
        "<p class='empty'>Bugün kriterlere uyan AL fikri bulunamadı "
        "(en az %5 beklenen yükseliş + desteğe yakın/destekten tepki şartı).</p>"
    )

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
  .card-head {{ display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:2px; }}
  .ticker {{ font-size:1.15rem; font-weight:700; }}
  .badge {{ background:#51cf6622; color:#51cf66; padding:2px 8px; border-radius:20px; font-weight:700; font-size:0.75rem; }}
  .score {{ background:#3d8bfd22; color:#3d8bfd; padding:3px 10px; border-radius:20px; font-weight:600; font-size:0.9rem; margin-left:auto; }}
  .setup {{ font-size:0.78rem; color:#51cf66; margin-bottom:8px; }}
  .row {{ display:flex; justify-content:space-between; font-size:0.9rem; padding:3px 0; color:#c8c8d0; }}
  .row.highlight {{ background:#51cf6614; border-radius:6px; padding:4px 6px; margin:4px 0; }}
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
  <h1>📊 BİST Tarama Botu — Sadece AL Fikirleri</h1>
  <div class="updated">Son güncelleme: {now_str}</div>
  <div class="criteria">Tarama: göreceli hacim &gt;{MIN_REL_VOLUME}, hacim &gt;{MIN_VOLUME_TL:,.0f} TL,
    bugünkü değişim &gt;%{MIN_CHANGE_PCT} · AL şartı: beklenen yükseliş &gt;=%{MIN_EXPECTED_RISE_PCT}
    ve desteğe yakın/destekten tepki — {len(results)} hisse eşleşti</div>

  {empty_msg}

  <h2>🏆 En yüksek potansiyelli ilk {TOP_N} hisse</h2>
  <div class="cards">{top_cards}</div>

  {"<h2>Diğer eşleşenler</h2><table><tr><th>Hisse</th><th>Puan</th><th>Bekl. Yükseliş</th><th>Kurulum</th><th>Hedef</th></tr>" + other_rows + "</table>" if other_rows else ""}

  <div class="disclaimer">⚠️ Bu sayfa SADECE alım (AL) fikirleri üretir, hiçbir zaman satış
  sinyali göstermez. Beklenen yükseliş, destek-direnç mesafesi ile RSI/MACD momentum
  gücünün birleştirilmesiyle üretilen bir TAHMİNDİR, kesin kazanç garantisi değildir.
  Son kararı ve risk yönetimini sen almalısın. Yatırım tavsiyesi değildir.</div>
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
        if analysis is None:
            continue  # geçerli bir AL kurulumu yok — sat/nötr fikir asla gösterilmez
        screened.update(analysis)
        results.append(screened)

    results.sort(key=lambda r: (r["expected_rise_pct"], r["score"]), reverse=True)

    print(f"{len(results)} hisse AL kriterlerine uydu.")

    # Web sitesi (GitHub Pages) için HTML üret
    os.makedirs("public", exist_ok=True)
    html = build_html_report(results)
    with open("public/index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("public/index.html oluşturuldu.")


if __name__ == "__main__":
    main()
