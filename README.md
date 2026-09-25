# Vysílačky

Skener PMR446 a LPD433 pro přijímač Airspy přes SoapySDR. Skript `vysilacky446.py` umožňuje vyhledávat aktivní kanály, poslouchat NBFM provoz a dekódovat CTCSS/DCS tóny.

## Funkce

- skenování 16 kanálů PMR446;
- volitelné skenování 69 kanálů LPD433;
- měření úrovně IQ signálu (RSSI);
- demodulace NBFM;
- hlasový filtr 300–3400 Hz;
- automatická regulace hlasitosti (AGC);
- adaptivní audio squelch;
- detekce CTCSS a DCS;
- přehrávání zvuku přes `sounddevice`;
- přímý poslech kanálu nebo libovolné frekvence.

## Požadavky

- Python 3
- přijímač Airspy
- nainstalovaný SoapySDR s Airspy ovladačem
- funkční audio výstup

Pythonové balíčky:

```bash
pip install SoapySDR numpy sounddevice
```

Na Linuxu může být potřeba nainstalovat SoapySDR a ovladač Airspy prostřednictvím balíčkovacího systému distribuce.

## Spuštění

Interaktivní skenování PMR446:

```bash
python3 vysilacky446.py
```

Přímý poslech PMR kanálu:

```bash
python3 vysilacky446.py --channel 8
```

Přímý poslech LPD kanálu:

```bash
python3 vysilacky446.py --lpd-channel 12
```

Přímý poslech frekvence v MHz:

```bash
python3 vysilacky446.py --freq 446.09375
```

Skenování PMR446 i LPD433:

```bash
python3 vysilacky446.py --lpd
```

Výpis dostupných audio zařízení:

```bash
python3 vysilacky446.py --list-devices
```

Výpis kanálů:

```bash
python3 vysilacky446.py --list
python3 vysilacky446.py --list --lpd
```

## Důležité parametry

| Parametr | Význam | Výchozí hodnota |
|---|---|---:|
| `--channel N` | přímý poslech PMR kanálu 1–16 | — |
| `--lpd-channel N` | přímý poslech LPD kanálu 1–69 | — |
| `--freq MHz` | přímý poslech zadané frekvence | — |
| `--lpd` | přidá kanály LPD433 ke skenování | vypnuto |
| `--rate N` | vzorkovací frekvence SDR | `2500000` |
| `--gain N` | RF zisk v dB | `30` |
| `--threshold N` | práh squelche nad šumovým dnem v dB | `1.0` |
| `--audio-gate N` | násobek audio šumového dna | `3.0` |
| `--dwell N` | doba měření jednoho kanálu v sekundách | `0.15` |
| `--device ZAŘÍZENÍ` | index nebo část názvu audio zařízení | automaticky |
| `--force-open` | testovací režim bez squelche | vypnuto |
| `--no-banner` | potlačí úvodní banner | vypnuto |

## Ovládání menu

- číslo nalezené stanice – začne poslech;
- `L` – přímé zadání PMR/LPD kanálu nebo frekvence;
- `Enter` nebo `s` – nové skenování;
- `q` – ukončení programu;
- `Ctrl+C` – návrat z poslechu do menu.

## Licence a použití

Používejte pouze frekvence a rádiový provoz v souladu s platnými právními předpisy a podmínkami místního telekomunikačního úřadu. Program je určen pro příjem rádiového provozu; před vysíláním ověřte příslušná oprávnění.

Autor: David Prošek
