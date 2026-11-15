# -*- coding: utf-8 -*-
"""
multifidus_unet.py  -  Multifidus Tez Asistani tarafindan uretildi.

Lumbar multifidus (LM) icin B-mod ultrason segmentasyonu: U-Net / U-Net++,
katilimci duzeyinde 5-kat capraz dogrulama + bagimsiz test seti.
Etik kurul protokolundeki yontemle uyumlu: Dice+BCE kaybi, Adam, erken durdurma,
yalnizca egitim verisine veri artirma, metrikler DSC, IoU, precision, recall, HD95.

KLASOR YAPISI
  VERI/B-mode/S-001_T0_ist_01.png      (ham goruntu)
  VERI/mask/S-001_T0_ist_01.png        (ayni ad; LM=255 ya da 1, arka plan=0)
  fold_atamalari.csv                    (uygulama > Faz 1 > Veri ayirma)

KURULUM
  pip install torch torchvision segmentation-models-pytorch pandas pillow scipy

KULLANIM
  python multifidus_unet.py --veri VERI --folds fold_atamalari.csv
  python multifidus_unet.py --veri VERI --folds fold_atamalari.csv --ogrenme_egrisi 10,20,30,40

CIKTILAR (sonuclar/ klasoru)
  segmentasyon_metrikleri.csv  -> uygulama > Faz 1 > Segmentasyon metrikleri > CSV ice aktar
  otomatik_olcumler.csv        -> uygulama > Faz 1 > Otomatik olcumleri ice aktar (H2)
  ogrenme_egrisi.csv           -> uygulama > Faz 1 > Ogrenme egrisi
  tahmin_maske/                -> modelin maskeleri (ayni dosya adlariyla)
  model_final.pt
"""
import argparse, os, random, math
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

PXMM_VARSAYILAN = 0        # piksel/mm (0 = mm cinsinden metrik verme)
EKRAN_SOLU_HASTA_SAGI = True      # transvers goruntude ekranin solu hastanin sagi mi?


# ----------------------------------------------------------------------------- yardimcilar
def katilimci_kodu(dosya):
    return os.path.basename(dosya).split("_")[0]


def maske_oku(yol):
    m = np.array(Image.open(yol).convert("L"))
    return (m > 0).astype(np.uint8)


def gri_oku(yol):
    return np.array(Image.open(yol).convert("L")).astype(np.float32)


def yuzey(m):
    m = m.astype(bool)
    if not m.any():
        return m
    return m & ~ndimage.binary_erosion(m, border_value=0)


def metrikler(gt, pr, pxmm=0.0):
    """gt, pr: 0/1 numpy dizileri (ayni boyut). DSC, IoU, precision, recall, HD95."""
    gt = gt.astype(bool); pr = pr.astype(bool)
    tp = np.logical_and(gt, pr).sum(); fp = np.logical_and(~gt, pr).sum(); fn = np.logical_and(gt, ~pr).sum()
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    sg, sp = yuzey(gt), yuzey(pr)
    if sg.any() and sp.any():
        dg = ndimage.distance_transform_edt(~sg)   # her pikselin gt yuzeyine uzakligi
        dp = ndimage.distance_transform_edt(~sp)
        hd = max(np.percentile(dg[sp], 95), np.percentile(dp[sg], 95))
        if pxmm and pxmm > 0:
            hd = hd / pxmm
    else:
        hd = float("nan")
    return dict(dice=dice, iou=iou, precision=prec, recall=rec, hd95=hd)


def taraf_olcumleri(maske, gri, pxmm):
    """Tek sinifli maskeyi sag/sol LM'ye ayirir; CSA (cm2) ve MGV hesaplar."""
    etiket, n = ndimage.label(maske)
    w = maske.shape[1]
    sol_ekran = np.zeros_like(maske, bool); sag_ekran = np.zeros_like(maske, bool)
    for i in range(1, n + 1):
        b = etiket == i
        if b.sum() < 20:          # gurultu
            continue
        cx = np.nonzero(b)[1].mean()
        if cx < w / 2:
            sol_ekran |= b
        else:
            sag_ekran |= b
    hasta_sag, hasta_sol = (sol_ekran, sag_ekran) if EKRAN_SOLU_HASTA_SAGI else (sag_ekran, sol_ekran)

    def olc(b):
        if not b.any():
            return float("nan"), float("nan")
        csa = b.sum() / (pxmm ** 2) / 100.0 if pxmm and pxmm > 0 else float("nan")
        return csa, float(gri[b].mean())
    csa_r, mgv_r = olc(hasta_sag); csa_l, mgv_l = olc(hasta_sol)
    return dict(csa_sag=csa_r, csa_sol=csa_l, mgv_sag=mgv_r, mgv_sol=mgv_l)


def veri_listesi(veri):
    bdir, mdir = os.path.join(veri, "B-mode"), os.path.join(veri, "mask")
    ok = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    satir = []
    for f in sorted(os.listdir(bdir)):
        if not f.lower().endswith(ok):
            continue
        mp = os.path.join(mdir, os.path.splitext(f)[0] + ".png")
        if not os.path.exists(mp):
            mp = os.path.join(mdir, f)
        satir.append(dict(dosya=f, goruntu=os.path.join(bdir, f), maske=mp if os.path.exists(mp) else None, kod=katilimci_kodu(f)))
    return pd.DataFrame(satir)


# ----------------------------------------------------------------------------- PyTorch kismi
def egitim_bilesenleri():
    import torch
    import torch.nn as nn
    import segmentation_models_pytorch as smp
    return torch, nn, smp


class LMVeri:
    def __init__(self, df, boyut, artir):
        self.df = df.reset_index(drop=True); self.boyut = boyut; self.artir = artir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        import torch
        r = self.df.iloc[i]
        g = Image.open(r.goruntu).convert("L"); m = Image.open(r.maske).convert("L")
        if self.artir:
            aci = random.uniform(-10, 10); olcek = random.uniform(0.9, 1.1)
            g = g.rotate(aci, resample=Image.BILINEAR); m = m.rotate(aci, resample=Image.NEAREST)
            if olcek != 1:
                W, H = g.size; nw, nh = int(W * olcek), int(H * olcek)
                g2, m2 = g.resize((nw, nh), Image.BILINEAR), m.resize((nw, nh), Image.NEAREST)
                g, m = Image.new("L", (W, H)), Image.new("L", (W, H))
                g.paste(g2, ((W - nw) // 2, (H - nh) // 2)); m.paste(m2, ((W - nw) // 2, (H - nh) // 2))
            if random.random() < 0.5:   # yatay yansitma (sag/sol LM ayni sinif)
                g = g.transpose(Image.FLIP_LEFT_RIGHT); m = m.transpose(Image.FLIP_LEFT_RIGHT)
        g = g.resize((self.boyut, self.boyut), Image.BILINEAR); m = m.resize((self.boyut, self.boyut), Image.NEAREST)
        x = np.array(g, np.float32) / 255.0
        if self.artir:
            x = np.clip((x - 0.5) * random.uniform(0.85, 1.15) + 0.5 + random.uniform(-0.1, 0.1), 0, 1)
        y = (np.array(m) > 0).astype(np.float32)
        return torch.from_numpy(x)[None], torch.from_numpy(y)[None]


def model_kur(mimari, onegitim):
    torch, nn, smp = egitim_bilesenleri()
    agirlik = "imagenet" if onegitim else None
    if mimari == "unetpp":
        return smp.UnetPlusPlus("resnet34", encoder_weights=agirlik, in_channels=1, classes=1)
    if mimari == "attunet":   # Attention U-Net'e yakin secenek: scSE dikkat bloklu U-Net
        return smp.Unet("resnet34", encoder_weights=agirlik, in_channels=1, classes=1, decoder_attention_type="scse")
    return smp.Unet("resnet34", encoder_weights=agirlik, in_channels=1, classes=1)


def tahmin(model, yol, boyut, cihaz):
    torch, _, _ = egitim_bilesenleri()
    g = Image.open(yol).convert("L"); W, H = g.size
    x = np.array(g.resize((boyut, boyut), Image.BILINEAR), np.float32) / 255.0
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(x)[None, None].to(cihaz)))[0, 0].cpu().numpy()
    p = np.array(Image.fromarray((p * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) / 255.0
    return (p > 0.5).astype(np.uint8)


def egit(egitim_df, dogrulama_df, a, cihaz, epoch_sabit=None):
    torch, nn, smp = egitim_bilesenleri()
    from torch.utils.data import DataLoader
    model = model_kur(a.mimari, a.onegitim).to(cihaz)
    dice_kayip = smp.losses.DiceLoss("binary"); bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    yukle = DataLoader(LMVeri(egitim_df, a.boyut, True), batch_size=a.batch, shuffle=True, num_workers=0)
    en_iyi, en_iyi_ep, bekle, durum = -1, 0, 0, None
    for ep in range(1, (epoch_sabit or a.epoch) + 1):
        model.train()
        for x, y in yukle:
            x, y = x.to(cihaz), y.to(cihaz); opt.zero_grad()
            out = model(x); kayip = dice_kayip(out, y) + bce(out, y); kayip.backward(); opt.step()
        if epoch_sabit:
            continue
        model.eval()
        d = np.mean([metrikler(maske_oku(r.maske), tahmin(model, r.goruntu, a.boyut, cihaz))["dice"] for r in dogrulama_df.itertuples()])
        if d > en_iyi:
            en_iyi, en_iyi_ep, bekle = d, ep, 0
            durum = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bekle += 1
            if bekle >= a.sabir:
                break
        print(f"  epoch {ep:3d}  dogrulama Dice {d:.4f}  (en iyi {en_iyi:.4f} @ {en_iyi_ep})")
    if durum is not None:
        model.load_state_dict(durum)
    return model, en_iyi_ep, en_iyi


# ----------------------------------------------------------------------------- ana akis
def main():
    ap = argparse.ArgumentParser(description="Lumbar multifidus U-Net segmentasyonu")
    ap.add_argument("--veri", required=True); ap.add_argument("--folds", required=True)
    ap.add_argument("--cikti", default="sonuclar"); ap.add_argument("--mimari", default="unet", choices=["unet", "unetpp", "attunet"])
    ap.add_argument("--boyut", type=int, default=256); ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--sabir", type=int, default=15, help="erken durdurma sabri (epoch)")
    ap.add_argument("--pxmm", type=float, default=PXMM_VARSAYILAN); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--onegitim", type=int, default=1, help="1: ImageNet on-egitimli kodlayici (internet gerekir)")
    ap.add_argument("--ogrenme_egrisi", default="", help="ornek: 10,20,30,40 (bagimsiz katilimci sayilari)")
    a = ap.parse_args()

    torch, _, _ = egitim_bilesenleri()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    cihaz = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    os.makedirs(os.path.join(a.cikti, "tahmin_maske"), exist_ok=True)

    df = veri_listesi(a.veri)
    eksik = df[df.maske.isna()]
    if len(eksik):
        print(f"UYARI: {len(eksik)} goruntunun maskesi yok, cikarildi.")
    df = df[df.maske.notna()]
    folds = pd.read_csv(a.folds); atama = dict(zip(folds.katilimci.astype(str), folds.altkume.astype(str)))
    df["altkume"] = df.kod.map(atama)
    if df.altkume.isna().any():
        print("UYARI: fold atamasi olmayan katilimcilar:", sorted(df[df.altkume.isna()].kod.unique()))
    df = df[df.altkume.notna()]
    gelistirme, test = df[df.altkume != "test"], df[df.altkume == "test"]
    k_listesi = sorted(gelistirme.altkume.unique())
    print(f"{df.kod.nunique()} katilimci, {len(df)} goruntu | gelistirme {len(gelistirme)} | test {len(test)} | cihaz {cihaz}")

    satirlar, olcumler, epoklar = [], [], []
    # 1) katilimci duzeyinde capraz dogrulama (out-of-fold tahminler)
    for k in k_listesi:
        print(f"\n== {k} dogrulama ==")
        tr, va = gelistirme[gelistirme.altkume != k], gelistirme[gelistirme.altkume == k]
        model, ep, _ = egit(tr, va, a, cihaz); epoklar.append(ep); model.eval()
        for r in va.itertuples():
            pr = tahmin(model, r.goruntu, a.boyut, cihaz)
            Image.fromarray(pr * 255).save(os.path.join(a.cikti, "tahmin_maske", os.path.splitext(r.dosya)[0] + ".png"))
            satirlar.append(dict(file=r.dosya, set="cv", fold=k.replace("fold", ""), **metrikler(maske_oku(r.maske), pr, a.pxmm)))
            olcumler.append(dict(file=r.dosya, **taraf_olcumleri(pr, gri_oku(r.goruntu), a.pxmm)))

    # 2) nihai model: tum gelistirme verisi, CV'deki medyan en iyi epoch kadar
    if len(test):
        n_ep = int(np.median(epoklar)) if epoklar else a.epoch
        print(f"\n== nihai model ({n_ep} epoch), bagimsiz test ==")
        model, _, _ = egit(gelistirme, None, a, cihaz, epoch_sabit=n_ep); model.eval()
        torch.save(model.state_dict(), os.path.join(a.cikti, "model_final.pt"))
        for r in test.itertuples():
            pr = tahmin(model, r.goruntu, a.boyut, cihaz)
            Image.fromarray(pr * 255).save(os.path.join(a.cikti, "tahmin_maske", os.path.splitext(r.dosya)[0] + ".png"))
            satirlar.append(dict(file=r.dosya, set="test", fold="test", **metrikler(maske_oku(r.maske), pr, a.pxmm)))
            olcumler.append(dict(file=r.dosya, **taraf_olcumleri(pr, gri_oku(r.goruntu), a.pxmm)))

    m = pd.DataFrame(satirlar); m.to_csv(os.path.join(a.cikti, "segmentasyon_metrikleri.csv"), index=False)
    pd.DataFrame(olcumler).to_csv(os.path.join(a.cikti, "otomatik_olcumler.csv"), index=False)
    for s in ["cv", "test"]:
        t = m[m.set == s]
        if len(t):
            print(f"{s}: " + "  ".join(f"{c} {t[c].mean():.3f}+-{t[c].std():.3f}" for c in ["dice", "iou", "precision", "recall", "hd95"]))

    # 3) ogrenme egrisi: sabit dogrulama kati, artan bagimsiz katilimci sayisi
    if a.ogrenme_egrisi:
        sabit = k_listesi[0]; va = gelistirme[gelistirme.altkume == sabit]; havuz = gelistirme[gelistirme.altkume != sabit]
        kodlar = list(havuz.kod.unique()); random.Random(a.seed).shuffle(kodlar); lc = []
        for n in [int(x) for x in a.ogrenme_egrisi.split(",") if x.strip()]:
            if n > len(kodlar):
                print(f"  {n} > havuz ({len(kodlar)}); atlandi"); continue
            tr = havuz[havuz.kod.isin(kodlar[:n])]
            model, _, _ = egit(tr, va, a, cihaz); model.eval()
            d = [metrikler(maske_oku(r.maske), tahmin(model, r.goruntu, a.boyut, cihaz))["dice"] for r in va.itertuples()]
            lc.append(dict(n=n, dice=float(np.mean(d)), sd=float(np.std(d, ddof=1)) if len(d) > 1 else 0.0)); print("  ogrenme egrisi", lc[-1])
        pd.DataFrame(lc).to_csv(os.path.join(a.cikti, "ogrenme_egrisi.csv"), index=False)
    print("\nBitti. Ciktilar:", os.path.abspath(a.cikti))


if __name__ == "__main__":
    main()
