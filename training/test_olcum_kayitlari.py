"""Olcum kayitlarinin bozuk girdide kapali kalma regresyonlari.

Rust olcumunu yeniden kosmaz; kayit sinirini bagimsiz olarak denetler.
Kosum: python3 -m unittest discover -s training -p test_olcum_kayitlari.py
"""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent


def yukle(ad):
    spec = importlib.util.spec_from_file_location(ad, ROOT / f"{ad}.py")
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


class OlcumKayitlari(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.moduller = [yukle(ad) for ad in ("kesit", "normalizasyon", "birlesik", "kademe")]

    def kayit(self, modul):
        return json.loads(modul.KAYIT.read_text(encoding="utf-8"))

    def test_mevcut_kayitlar_gecer(self):
        for modul in self.moduller:
            with self.subTest(modul=modul.__name__):
                self.assertIsNone(modul._bulgu(self.kayit(modul)))

    def test_kok_ve_kanit_nesne_olmali(self):
        for modul in self.moduller:
            for bozuk in (None, [], True, 42, "kanit"):
                with self.subTest(modul=modul.__name__, bozuk=bozuk):
                    self.assertIsNotNone(modul._bulgu(bozuk))
                    kayit = self.kayit(modul)
                    kayit["kanit"] = bozuk
                    self.assertIsNotNone(modul._bulgu(kayit))

    def test_her_alanda_tur_ve_sonluluk(self):
        for modul in self.moduller:
            temiz = self.kayit(modul)
            alanlar = modul.TAM_ALANLAR + getattr(modul, "KESIRLI_ALANLAR", ())
            for alan in alanlar:
                for bozuk in (True, False, None, "1", [], {}, float("nan"), float("inf"), -float("inf")):
                    with self.subTest(modul=modul.__name__, alan=alan, bozuk=bozuk):
                        kayit = copy.deepcopy(temiz)
                        kayit["kanit"][alan] = bozuk
                        self.assertIsNotNone(modul._bulgu(kayit))

    def test_sayimlar_pozitif_tam_sayi(self):
        for modul in self.moduller:
            for alan in modul.TAM_ALANLAR:
                for bozuk in (-1, 0, 0.5, 1.5):
                    with self.subTest(modul=modul.__name__, alan=alan, bozuk=bozuk):
                        kayit = self.kayit(modul)
                        kayit["kanit"][alan] = bozuk
                        self.assertIsNotNone(modul._bulgu(kayit))

    def test_eksik_alanlar_reddedilir(self):
        for modul in self.moduller:
            alanlar = modul.TAM_ALANLAR + getattr(modul, "KESIRLI_ALANLAR", ())
            for alan in alanlar:
                kayit = self.kayit(modul)
                del kayit["kanit"][alan]
                self.assertIsNotNone(modul._bulgu(kayit))

    def test_norm_negatif_buyukluk_reddedilir(self):
        modul = self.moduller[1]
        for alan in ("merkez_fark", "duz_rms_fark", "gradyan_sapma", "cikti_rms2"):
            kayit = self.kayit(modul)
            kayit["kanit"][alan] = -1
            self.assertIsNotNone(modul._bulgu(kayit))

    def test_norm_ortalama_isaretten_bagimsiz(self):
        modul = self.moduller[1]
        for ortalama in (-0.5, 0.5):
            kayit = self.kayit(modul)
            kayit["kanit"]["cikti_ort"] = ortalama
            self.assertIsNotNone(modul._bulgu(kayit))

    def test_norm_rms_ust_siniri(self):
        modul = self.moduller[1]
        kayit = self.kayit(modul)
        kayit["kanit"]["cikti_rms2"] = 1.1
        self.assertIsNotNone(modul._bulgu(kayit))

    def test_olcut_turu_ve_celiski(self):
        for modul in self.moduller:
            for bozuk in (None, 1, "true", False):
                kayit = self.kayit(modul)
                kayit["olcut"]["sonuc"] = bozuk
                self.assertIsNotNone(modul._bulgu(kayit))


if __name__ == "__main__":
    unittest.main()
