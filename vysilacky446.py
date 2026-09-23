#!/usr/bin/env python3
"""
spy.py - Stanice Davida Proška
================================
Skener vysilacek pro Airspy (pres SoapySDR)

Skenuje PMR446 / LPD433, meri IQ vykon (RSSI), vykresli seznam
stanic. Po vyberu stanice:
  - demoduluje NBFM
  - filtruje hlasove pasmo 300-3400 Hz
  - AGC normalizuje hlasitost
  - ADAPTIVNI AUDIO SQUELCH (RMS hlasoveho pasma)
  - dekoduje CTCSS a DCS

Prime poslouchani:
  - --channel N         (PMR 1-16)
  - --lpd-channel N     (LPD 1-69)
  - --freq MHZ          (libovolny kmitocet)
  - nebo v menu volba L / primo zadanim (napr. "pmr8", "446.09375")

Zavislosti:
    pip install SoapySDR numpy sounddevice

Pouziti:
    python3 spy.py                       # interaktivni rezim
    python3 spy.py --channel 8           # rovnou poslouchat PMR8
    python3 spy.py --freq 446.09375      # rovnou poslouchat dany MHz
    python3 spy.py --threshold 1.0       # citlivejsi prah
    python3 spy.py --audio-gate 2.5      # faktor audio squelche
    python3 spy.py --force-open          # pro test - vzdy otevreno
"""

import argparse
import queue
import sys
import time

import numpy as np

try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
except ImportError:
    sys.exit("Chyba: neni nainstalovan modul 'SoapySDR' (pip install SoapySDR)")

try:
    import sounddevice as sd
except ImportError:
    sd = None


# ===========================================================================
# IDENTITA STANICE
# ===========================================================================
STATION_NAME = "Stanice Davida Proška"
STATION_SHORT = "SDP"
STATION_VERSION = "1.0"


def print_banner():
    """Vypise uvodni banner stanice."""
    line = "=" * 66
    print()
    print(line)
    print("   ####  #####  #####")
    print("  #      #   #  #   #     %s" % STATION_NAME)
    print("   ###   #   #  #####     verze %s" % STATION_VERSION)
    print("      #  #   #  #")
    print("  ####   #####  #         Airspy PMR446 / LPD433 skener")
    print(line)
    print("   NBFM | hlasove pasmo 300-3400 Hz | AGC | CTCSS + DCS")
    print(line)
    print()


def print_footer():
    """Vypise rozluckovy text."""
    print()
    print("=" * 66)
    print("  73 de %s (%s)" % (STATION_NAME, STATION_SHORT))
    print("=" * 66)
    print()


# ===========================================================================
# Kanaly
# ===========================================================================
PMR446 = [446006250 + i * 12500 for i in range(16)]
LPD433 = [433075000 + i * 25000 for i in range(69)]


def make_channels(pmr=True, lpd=False):
    chans = []
    if pmr:
        chans += [("PMR%d" % (i + 1), f) for i, f in enumerate(PMR446)]
    if lpd:
        chans += [("LPD%d" % (i + 1), f) for i, f in enumerate(LPD433)]
    return chans


# ===========================================================================
# CTCSS / DCS tabulky
# ===========================================================================
CTCSS_TONES = [
    67.0, 69.3, 71.9, 74.4, 77.0, 79.7, 82.5, 85.4, 88.5, 91.5,
    94.8, 97.4, 100.0, 103.5, 107.2, 110.9, 114.8, 118.8, 123.0,
    127.3, 131.8, 136.5, 141.3, 146.2, 151.4, 156.7, 162.2, 167.9,
    173.8, 179.9, 186.2, 192.8, 203.5, 210.7, 218.1, 225.7, 233.6,
    241.8, 250.3,
]

DCS_CODES = [
    "023", "025", "026", "031", "032", "036", "043", "047", "051", "053",
    "054", "065", "071", "072", "073", "074", "114", "115", "116", "122",
    "125", "131", "132", "134", "143", "145", "152", "155", "156", "162",
    "165", "172", "174", "205", "212", "223", "225", "226", "243", "244",
    "245", "246", "251", "252", "255", "261", "263", "265", "266", "271",
    "274", "306", "311", "315", "325", "331", "332", "343", "346", "351",
    "356", "364", "365", "371", "411", "412", "413", "423", "431", "432",
    "445", "446", "452", "454", "455", "462", "464", "465", "466", "503",
    "506", "516", "523", "526", "532", "546", "565", "606", "612", "624",
    "627", "631", "632", "654", "662", "664", "703", "712", "723", "731",
    "732", "734", "743", "754",
]

DCS_CARRIER_HZ = 134.4
DCS_DEVIATION_HZ = 2.4
DCS_BIT_RATE = 134.4


# ===========================================================================
# Goertzel
# ===========================================================================
def goertzel_power(x, freq, fs):
    n = len(x)
    if n == 0:
        return 0.0
    t = np.arange(n)
    phase = 2.0 * np.pi * freq * t / fs
    i = float(np.dot(x, np.cos(phase)))
    q = float(np.dot(x, np.sin(phase)))
    return i * i + q * q


# ===========================================================================
# Biquad + VoiceBandFilter + AGC
# ===========================================================================
class Biquad:
    def __init__(self, b, a):
        self.b = np.array(b, dtype=np.float64)
        self.a = np.array(a, dtype=np.float64)
        self.z1 = 0.0
        self.z2 = 0.0

    def process(self, x):
        x = np.asarray(x, dtype=np.float64)
        y = np.empty_like(x)
        b0, b1, b2 = self.b
        a1, a2 = self.a[1], self.a[2]
        z1, z2 = self.z1, self.z2
        for i in range(len(x)):
            xi = x[i]
            yi = b0 * xi + z1
            z1 = b1 * xi - a1 * yi + z2
            z2 = b2 * xi - a2 * yi
            y[i] = yi
        self.z1 = z1
        self.z2 = z2
        return y

    def reset(self):
        self.z1 = 0.0
        self.z2 = 0.0


def make_hpf(fs, f0, q=0.7071):
    w0 = 2.0 * np.pi * f0 / fs
    cw = np.cos(w0)
    alpha = np.sin(w0) / (2.0 * q)
    b = [(1.0 + cw) / 2.0, -(1.0 + cw), (1.0 + cw) / 2.0]
    a = [1.0 + alpha, -2.0 * cw, 1.0 - alpha]
    return ([v / a[0] for v in b], [1.0, a[1] / a[0], a[2] / a[0]])


def make_lpf(fs, f0, q=0.7071):
    w0 = 2.0 * np.pi * f0 / fs
    cw = np.cos(w0)
    alpha = np.sin(w0) / (2.0 * q)
    b = [(1.0 - cw) / 2.0, 1.0 - cw, (1.0 - cw) / 2.0]
    a = [1.0 + alpha, -2.0 * cw, 1.0 - alpha]
    return ([v / a[0] for v in b], [1.0, a[1] / a[0], a[2] / a[0]])


class VoiceBandFilter:
    def __init__(self, fs, low=300.0, high=3400.0):
        self.hpf = Biquad(*make_hpf(fs, low))
        self.lpf = Biquad(*make_lpf(fs, high))

    def process(self, x):
        return self.lpf.process(self.hpf.process(x)).astype(np.float32)

    def reset(self):
        self.hpf.reset()
        self.lpf.reset()


class AGC:
    def __init__(self, target=0.3, attack=0.01, decay=0.0005,
                 max_gain=30.0, min_gain=0.5):
        self.target = target
        self.attack = attack
        self.decay = decay
        self.max_gain = max_gain
        self.min_gain = min_gain
        self.gain = 1.0

    def process(self, x):
        if len(x) == 0:
            return x
        level = float(np.sqrt(np.mean(x * x))) + 1e-9
        desired = float(np.clip(self.target / level,
                                self.min_gain, self.max_gain))
        if desired < self.gain:
            self.gain = (1.0 - self.attack) * self.gain + self.attack * desired
        else:
            self.gain = (1.0 - self.decay) * self.gain + self.decay * desired
        return np.clip(x * self.gain, -1.0, 1.0).astype(np.float32)

    def reset(self):
        self.gain = 1.0


# ===========================================================================
# Adaptivni audio squelch
# ===========================================================================
class AudioSquelch:
    def __init__(self, floor_init=0.02, factor=3.0,
                 floor_attack=0.20, floor_decay=0.0005,
                 hang_frames=25, min_floor=1e-4):
        self.audio_floor = float(max(floor_init, min_floor))
        self.factor = float(factor)
        self.floor_attack = float(floor_attack)
        self.floor_decay = float(floor_decay)
        self.hang_frames = int(hang_frames)
        self.min_floor = float(min_floor)
        self.is_open = False
        self.hang = 0

    def process(self, rms_value):
        if not self.is_open:
            if rms_value < self.audio_floor:
                self.audio_floor = ((1.0 - self.floor_attack) * self.audio_floor
                                    + self.floor_attack * rms_value)
            else:
                self.audio_floor = ((1.0 - self.floor_decay) * self.audio_floor
                                    + self.floor_decay * rms_value)
        if self.audio_floor < self.min_floor:
            self.audio_floor = self.min_floor
        threshold = self.audio_floor * self.factor
        detected = rms_value > threshold
        if detected:
            self.is_open = True
            self.hang = self.hang_frames
        elif self.hang > 0:
            self.hang -= 1
        else:
            self.is_open = False
        return self.is_open, self.audio_floor, threshold

    def reset(self):
        self.is_open = False
        self.hang = 0


# ===========================================================================
# CTCSS
# ===========================================================================
class CTCSSDecoder:
    def __init__(self, sample_rate, block_size=4096):
        self.fs = float(sample_rate)
        self.block_size = int(block_size)
        self.buffer = np.zeros(0, dtype=np.float32)
        self.current_tone = None
        self.tone_history = []
        self.min_stable_blocks = 3

    def process(self, audio):
        self.buffer = np.concatenate((self.buffer, audio))
        while len(self.buffer) >= self.block_size:
            block = self.buffer[:self.block_size]
            self.buffer = self.buffer[self.block_size:]
            powers = np.array([goertzel_power(block, f, self.fs)
                               for f in CTCSS_TONES])
            idx = int(np.argmax(powers))
            max_p = powers[idx]
            total_p = float(np.sum(powers)) + 1e-12
            conf = max_p / total_p
            tone = CTCSS_TONES[idx] if (conf > 0.05 and max_p > 1e-6) else None
            if tone is not None:
                self.tone_history.append(tone)
                if len(self.tone_history) > self.min_stable_blocks:
                    self.tone_history.pop(0)
                if len(self.tone_history) >= self.min_stable_blocks:
                    vals = np.array(self.tone_history)
                    if np.std(vals) < 1.0:
                        self.current_tone = int(round(float(np.mean(vals))))
            else:
                self.tone_history.clear()
                self.current_tone = None
        return self.current_tone


# ===========================================================================
# DCS
# ===========================================================================
class DCSDecoder:
    F_MARK = DCS_CARRIER_HZ + DCS_DEVIATION_HZ
    F_SPACE = DCS_CARRIER_HZ - DCS_DEVIATION_HZ
    BIT_RATE = DCS_BIT_RATE
    GOLAY_POLY = 0xC75

    def __init__(self, sample_rate):
        self.fs = float(sample_rate)
        self.samples_per_bit = self.fs / self.BIT_RATE
        self.sample_buffer = np.zeros(0, dtype=np.float32)
        self.bits = []
        self.MAX_BITS = 92
        self.current_code = None
        self.stable_count = 0

    def _detect_bit(self, block):
        return 1 if goertzel_power(block, self.F_MARK, self.fs) > \
                    goertzel_power(block, self.F_SPACE, self.fs) else 0

    def _golay_syndrome(self, word23):
        m = (word23 >> 11) & 0xFFF
        r = word23 & 0x7FF
        reg = m << 11
        for i in range(22, 10, -1):
            if reg & (1 << i):
                reg ^= self.GOLAY_POLY << (i - 11)
        return r ^ (reg & 0x7FF)

    def _golay_ok(self, word23):
        syn = self._golay_syndrome(word23)
        return syn == 0 or bin(syn).count("1") <= 1

    def _bits_to_code(self, b9):
        if len(b9) != 9:
            return None
        u = b9[0] | (b9[1] << 1) | (b9[2] << 2)
        t = b9[3] | (b9[4] << 1) | (b9[5] << 2)
        h = b9[6] | (b9[7] << 1) | (b9[8] << 2)
        if u > 7 or t > 7 or h > 7:
            return None
        return "%d%d%d" % (h, t, u)

    def process(self, audio):
        if len(audio) == 0:
            return self.current_code if self.stable_count >= 2 else None
        self.sample_buffer = np.concatenate((self.sample_buffer, audio))
        n_bit = int(round(self.samples_per_bit))
        if n_bit < 4:
            return None
        while len(self.sample_buffer) >= n_bit:
            block = self.sample_buffer[:n_bit]
            self.sample_buffer = self.sample_buffer[n_bit:]
            self.bits.append(self._detect_bit(block))
            if len(self.bits) > self.MAX_BITS:
                self.bits.pop(0)
        if len(self.bits) < 23:
            return None
        found = None
        for off in range(len(self.bits) - 23 + 1):
            wb = self.bits[off:off + 23]
            if wb[0] != 1 or wb[10] != 0:
                continue
            w = 0
            for b in wb:
                w = (w << 1) | b
            if not self._golay_ok(w):
                continue
            code = self._bits_to_code(wb[1:10])
            if code and code in DCS_CODES:
                found = code
                break
        if found is not None:
            if found == self.current_code:
                self.stable_count += 1
            else:
                self.current_code = found
                self.stable_count = 1
        else:
            if self.stable_count > 0:
                self.stable_count -= 1
            if self.stable_count == 0:
                self.current_code = None
        if self.stable_count >= 2 and self.current_code is not None:
            return self.current_code
        return None

    def reset(self):
        self.sample_buffer = np.zeros(0, dtype=np.float32)
        self.bits = []
        self.current_code = None
        self.stable_count = 0


# ===========================================================================
# Audio device
# ===========================================================================
HDMI_MARKERS = ("hdmi", "displayport", "dp-", "dp ", "nvidia", "gpu", "eld")
JACK_MARKERS = ("analog", "speaker", "headphone", "headset", "jack",
                "sluchatka", "repro", "line out", "lineout", "front")


def find_audio_device(prefer=None):
    if sd is None:
        return None
    devices = sd.query_devices()
    if prefer is not None:
        try:
            idx = int(prefer)
            if 0 <= idx < len(devices) and devices[idx]["max_output_channels"] > 0:
                return idx
        except (ValueError, TypeError):
            pass
        pn = str(prefer).lower()
        for idx, d in enumerate(devices):
            if d["max_output_channels"] > 0 and pn in d["name"].lower():
                return idx
        print("[!] Audio zarizeni '%s' nenalezeno, zvolim automaticky." % prefer)
    for idx, d in enumerate(devices):
        if d["max_output_channels"] <= 0:
            continue
        name = d["name"].lower()
        if any(m in name for m in HDMI_MARKERS):
            continue
        if any(m in name for m in JACK_MARKERS):
            return idx
    for idx, d in enumerate(devices):
        if d["max_output_channels"] <= 0:
            continue
        if any(m in d["name"].lower() for m in HDMI_MARKERS):
            continue
        return idx
    try:
        return sd.default.device[1]
    except Exception:
        return None


def describe_device(idx):
    if sd is None or idx is None:
        return "(zadne)"
    try:
        return "#%d %s" % (idx, sd.query_devices(idx)["name"])
    except Exception:
        return "#%s" % idx


# ===========================================================================
# SDR - pojmenovany prijimac Stanice Davida Proška
# ===========================================================================
class AirspyReceiver:
    """
    Prijimac Airspy provozovany Stanici Davida Proška.
    Pri inicializaci se v logu identifikuje jako stanice SDP.
    """

    def __init__(self, sample_rate=2.5e6, gain=30):
        print("[%s] Hledam Airspy..." % STATION_SHORT)
        results = SoapySDR.Device.enumerate({"driver": "airspy"})
        if not results:
            sys.exit("[%s] Airspy nenalezen! Zkontroluj USB a driver."
                     % STATION_SHORT)

        self.sdr = SoapySDR.Device(results[0])
        hw = self.sdr.getHardwareKey()
        print("[%s] Prijimac pripraven: %s" % (STATION_SHORT, hw))
        print("[%s] Zarizeni je provozovano jako '%s'"
              % (STATION_SHORT, STATION_NAME))

        # Pokus o nastaveni uzivatelskeho nazvu (pokud driver podporuje)
        for key in ("label", "name", "device_name", "user_label"):
            try:
                self.sdr.writeSetting(key, STATION_NAME)
                print("[%s] Nastaveno '%s' = '%s'"
                      % (STATION_SHORT, key, STATION_NAME))
            except Exception:
                pass

        self.rate = float(sample_rate)
        self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self.rate)
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, 435e6)
        self.sdr.setGainMode(SOAPY_SDR_RX, 0, False)
        self.sdr.setGain(SOAPY_SDR_RX, 0, gain)
        try:
            self.sdr.setBandwidth(SOAPY_SDR_RX, 0, min(self.rate, 1e6))
        except Exception:
            pass

        self.rx_stream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        self.sdr.activateStream(self.rx_stream)
        self.buf_len = int(self.rate * 0.02)
        self.buff = np.zeros(self.buf_len * 2, dtype=np.float32)

        print("[%s] Vzorkovaci frekvence: %.3f MHz, zisk: %.1f dB"
              % (STATION_SHORT, self.rate / 1e6, gain))
        print("[%s] Stream aktivni, pripraven ke skenu." % STATION_SHORT)
        print()

    def set_frequency(self, freq_hz):
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, freq_hz)

    def read_iq(self):
        sr = self.sdr.readStream(self.rx_stream, [self.buff],
                                 self.buf_len, timeoutUs=500000)
        if sr.ret < 0:
            return None
        raw = self.buff[: sr.ret * 2]
        return raw[0::2] + 1j * raw[1::2]

    def station_id(self):
        return "%s (%s)" % (STATION_NAME, STATION_SHORT)


# ===========================================================================
# NBFM
# ===========================================================================
class NbfmDemod:
    def __init__(self, in_rate, out_rate=48000, deviation=3000):
        self.in_rate = float(in_rate)
        self.out_rate = float(out_rate)
        self.deviation = float(deviation)
        self.last_phase = 0.0
        self.decim = int(in_rate // out_rate)
        assert self.decim > 0
        self.deemph_a = np.exp(-1.0 / (self.out_rate * 750e-6))
        self.deemph_y = 0.0

    def reset(self):
        self.last_phase = 0.0
        self.deemph_y = 0.0

    def demodulate(self, iq):
        phase = np.angle(iq)
        dphase = np.diff(phase, prepend=self.last_phase)
        self.last_phase = phase[-1]
        dphase = (dphase + np.pi) % (2 * np.pi) - np.pi
        audio = dphase * self.in_rate / (2 * np.pi * self.deviation)
        n = len(audio) // self.decim * self.decim
        if n == 0:
            return None
        audio = audio[:n].reshape(-1, self.decim).mean(axis=1)
        y = np.empty_like(audio)
        prev = self.deemph_y
        a = self.deemph_a
        for i, x in enumerate(audio):
            prev = x + a * (prev - x)
            y[i] = prev
        self.deemph_y = prev
        return y.astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


# ===========================================================================
# Skener
# ===========================================================================
def scan_pass(receiver, channels, dwell, margin_db=1.0, verbose=True):
    measurements = []
    total = len(channels)
    if verbose:
        print("-" * 66)
        print(" [%s] Skenuji %d kanalu..." % (STATION_SHORT, total))
        print("-" * 66)
        print("  #   kanal    kmitocet      RSSI      stav")
        print("-" * 66)

    for i, (name, freq) in enumerate(channels, 1):
        receiver.set_frequency(freq)
        time.sleep(0.025)
        blok_cas = receiver.buf_len / receiver.rate
        n_bloku = max(2, int(dwell / blok_cas))
        vals = []
        for _ in range(n_bloku):
            iq = receiver.read_iq()
            if iq is not None and len(iq):
                vals.append(float(np.mean(np.abs(iq) ** 2)))
        p_db = 10.0 * np.log10(float(np.mean(vals)) + 1e-20) if vals else -120.0
        measurements.append((name, freq, p_db))
        if verbose:
            sys.stdout.write("  %2d  %-6s  %10.5f MHz  %6.1f dB   ...\r" %
                             (i, name, freq / 1e6, p_db))
            sys.stdout.flush()

    if verbose:
        sys.stdout.write("\n")
    valid = [p for (_, _, p) in measurements]
    noise_floor = float(np.percentile(valid, 20)) if valid else -100.0
    threshold_db = noise_floor + margin_db

    if verbose:
        print("-" * 66)
        for i, (name, freq, p_db) in enumerate(measurements, 1):
            delta = p_db - noise_floor
            stav = "  *** SIGNAL ***  +%.1f dB" % delta if p_db > threshold_db \
                   else "  ---  +%.1f dB" % delta
            print("  %2d  %-6s  %10.5f MHz  %6.1f dB%s" %
                  (i, name, freq / 1e6, p_db, stav))
        print("-" * 66)
        print("[%s] Sumove dno IQ: %.1f dB, prah: %.1f dB (+%.1f dB)" %
              (STATION_SHORT, noise_floor, threshold_db, margin_db))

    found = [(name, freq, p_db - noise_floor)
             for (name, freq, p_db) in measurements if p_db > threshold_db]
    return found, noise_floor, measurements


# ===========================================================================
# Poslech jednoho kanalu
# ===========================================================================
def listen_single(receiver, demod, name, freq, audio_device=None,
                  squelch_db=1.0, audio_factor=3.0, force_open=False):
    if sd is None:
        sys.exit("sounddevice potreba pro poslech (pip install sounddevice)")

    receiver.set_frequency(freq)
    demod.reset()
    time.sleep(0.05)

    voice = VoiceBandFilter(demod.out_rate, low=300.0, high=3400.0)

    print("[%s] Kalibruji sumove dno (IQ i audio)..." % STATION_SHORT)
    calib_iq = []
    calib_audio = []
    for _ in range(10):
        iq = receiver.read_iq()
        if iq is None or not len(iq):
            continue
        p = float(np.mean(np.abs(iq) ** 2))
        calib_iq.append(10.0 * np.log10(p + 1e-20))
        a = demod.demodulate(iq)
        if a is not None:
            va = voice.process(a)
            calib_audio.append(rms(va))

    noise_floor_iq = float(np.median(calib_iq)) if calib_iq else -90.0
    threshold_iq_db = noise_floor_iq + squelch_db
    audio_floor_init = (float(np.percentile(calib_audio, 20))
                        if calib_audio else 0.02)
    audio_floor_init = max(audio_floor_init, 1e-4)
    print("[%s]   IQ sum ~ %.1f dB (prah %.1f dB)" %
          (STATION_SHORT, noise_floor_iq, threshold_iq_db))
    print("[%s]   audio sum ~ %.4f (audio squelch: RMS > %.4f * floor)" %
          (STATION_SHORT, audio_floor_init, audio_factor))

    q = queue.Queue(maxsize=64)
    ctcss = CTCSSDecoder(sample_rate=demod.out_rate)
    dcs = DCSDecoder(sample_rate=demod.out_rate)
    agc = AGC(target=0.35, attack=0.02, decay=0.001,
              max_gain=40.0, min_gain=0.5)
    audio_sq = AudioSquelch(floor_init=audio_floor_init,
                            factor=audio_factor,
                            hang_frames=25)

    last_ctcss = None
    last_dcs = None

    def cb(outdata, frames, t, status):
        try:
            data = q.get_nowait()
        except queue.Empty:
            data = np.zeros(frames, dtype=np.float32)
        n = min(len(data), frames)
        outdata[:n, 0] = data[:n]
        if n < frames:
            outdata[n:, 0] = 0

    try:
        stream = sd.OutputStream(
            samplerate=int(demod.out_rate), channels=1,
            dtype="float32", callback=cb, device=audio_device,
        )
    except Exception as e:
        print("[!] Nelze otevrit audio vystup (%s): %s" %
              (describe_device(audio_device), e))
        return

    with stream:
        print()
        print("=" * 66)
        print(" %s" % STATION_NAME)
        print(" POSLECH %s @ %.5f MHz" % (name, freq / 1e6))
        print(" Hlasove pasmo 300-3400 Hz | AGC | AUDIO squelch | CTCSS + DCS")
        if force_open:
            print(" [!] --force-open: audio jde vzdy ven (bez squelche)")
        print(" Ctrl+C = zpet do menu")
        print("=" * 66)
        try:
            while True:
                iq = receiver.read_iq()
                if iq is None:
                    continue

                p_db = 10.0 * np.log10(
                    float(np.mean(np.abs(iq) ** 2)) + 1e-20)
                if p_db < noise_floor_iq:
                    noise_floor_iq = 0.90 * noise_floor_iq + 0.10 * p_db
                else:
                    noise_floor_iq = 0.9995 * noise_floor_iq + 0.0005 * p_db
                threshold_iq_db = noise_floor_iq + squelch_db
                carrier_iq = p_db > threshold_iq_db

                audio = demod.demodulate(iq)
                if audio is None:
                    continue

                tone = ctcss.process(audio)
                if tone != last_ctcss:
                    if tone is not None:
                        print("\n   >>> CTCSS: %.1f Hz" % tone)
                    last_ctcss = tone

                code = dcs.process(audio)
                if code != last_dcs:
                    if code is not None:
                        print("\n   >>> DCS:   %s (octal)" % code)
                    last_dcs = code

                voice_audio = voice.process(audio)
                vlevel = rms(voice_audio)

                audio_open, audio_floor, audio_thr = audio_sq.process(vlevel)

                if force_open:
                    squelch_open = True
                else:
                    squelch_open = audio_open or carrier_iq

                if squelch_open:
                    out = agc.process(voice_audio)
                    out = np.clip(out * 1.5, -1.0, 1.0).astype(np.float32)
                else:
                    agc.reset()
                    out = np.zeros_like(voice_audio)

                while q.full():
                    q.get_nowait()
                q.put(out)

                status = " [%s] RX %-6s IQ%+5.1fdB audio=%.3f/%.3f [%s]" % (
                    STATION_SHORT, name, p_db - noise_floor_iq,
                    vlevel, audio_thr,
                    "OPEN" if squelch_open else "mute")
                if last_ctcss is not None:
                    status += " CTCSS=%.1fHz" % last_ctcss
                if last_dcs is not None:
                    status += " DCS=%s" % last_dcs
                sys.stdout.write(status + "   \r")
                sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n[%s] Ukoncuji poslech..." % STATION_SHORT)


# ===========================================================================
# Prime poslouchani
# ===========================================================================
def resolve_station(spec):
    s = spec.strip()
    if not s:
        return None
    up = s.upper()

    if up.startswith("PMR"):
        try:
            n = int(up[3:])
        except ValueError:
            return None
        if 1 <= n <= 16:
            return ("PMR%d" % n, PMR446[n - 1])
        return None

    if up.startswith("LPD"):
        try:
            n = int(up[3:])
        except ValueError:
            return None
        if 1 <= n <= 69:
            return ("LPD%d" % n, LPD433[n - 1])
        return None

    try:
        n = int(s)
        if 1 <= n <= 16:
            return ("PMR%d" % n, PMR446[n - 1])
        if 17 <= n <= 85:
            return ("LPD%d" % (n - 16), LPD433[n - 17])
        return None
    except ValueError:
        pass

    try:
        mhz = float(s.replace(",", "."))
        hz = int(round(mhz * 1e6))
        if 1_000_000 <= hz <= 6_000_000_000:
            return ("%.5f MHz" % (hz / 1e6), hz)
        return None
    except ValueError:
        return None


def listen_direct_menu(receiver, demod, audio_device=None,
                       squelch_db=1.0, audio_factor=3.0, force_open=False):
    print()
    print("=" * 66)
    print(" %s - PRIME POSLOUCHANI STANICE" % STATION_NAME)
    print("=" * 66)
    print("   1-16            = PMR kanal")
    print("   PMR1..PMR16     = PMR kanal")
    print("   LPD1..LPD69     = LPD kanal")
    print("   <MHz>           = libovolny kmitocet (napr. 446.09375)")
    print("   Enter           = zpet")
    print("=" * 66)
    try:
        spec = input("Stanice: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if not spec:
        return
    res = resolve_station(spec)
    if res is None:
        print("Nerozumim zadani '%s'." % spec)
        return
    name, freq = res
    print("--> %s @ %.5f MHz" % (name, freq / 1e6))
    listen_single(receiver, demod, name, freq,
                  audio_device=audio_device, squelch_db=squelch_db,
                  audio_factor=audio_factor, force_open=force_open)


# ===========================================================================
# Hlavni menu
# ===========================================================================
def scan(receiver, demod, channels, dwell=0.15, margin_db=1.0,
         audio_device=None, audio_factor=3.0, force_open=False):
    while True:
        print()
        print(">>> %s - SKENUJI %d KANALU <<<"
              % (STATION_NAME.upper(), len(channels)))
        found, noise_floor, all_meas = scan_pass(
            receiver, channels, dwell, margin_db=margin_db, verbose=True)

        print()
        print("=" * 66)
        if found:
            print(" %s - NALEZENE STANICE (nad prahem %.1f dB):"
                  % (STATION_SHORT, margin_db))
            for i, (name, freq, delta) in enumerate(found, 1):
                print("  %2d)  %-6s  %10.5f MHz   +%.1f dB" %
                      (i, name, freq / 1e6, delta))
        else:
            print(" [%s] Zadny signal nad prahem (zkus --threshold 0.5)"
                  % STATION_SHORT)
        print("=" * 66)
        print("  <cislo>       = poslouchat nalezenou stanici")
        print("  L             = primo zadat stanici (PMR/LPD/MHz)")
        print("  Enter nebo s  = znovu skenovat")
        print("  q             = konec")
        print("=" * 66)

        try:
            choice = input("[%s] Volba: " % STATION_SHORT).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        low = choice.lower()
        if low in ("q", "quit", "exit"):
            return
        if low in ("", "s", "scan"):
            continue
        if low in ("l", "listen"):
            listen_direct_menu(receiver, demod,
                               audio_device=audio_device,
                               squelch_db=margin_db,
                               audio_factor=audio_factor,
                               force_open=force_open)
            continue

        res = resolve_station(choice)
        if res is not None:
            name, freq = res
            print("--> [%s] primo poslouchat %s @ %.5f MHz" %
                  (STATION_SHORT, name, freq / 1e6))
            listen_single(receiver, demod, name, freq,
                          audio_device=audio_device, squelch_db=margin_db,
                          audio_factor=audio_factor, force_open=force_open)
            continue

        try:
            idx = int(choice) - 1
        except ValueError:
            print("Nerozumim volbe '%s'." % choice)
            continue

        if not 0 <= idx < len(found):
            print("Neplatna volba (mimo rozsah 1-%d)." % len(found))
            continue

        name, freq, _ = found[idx]
        listen_single(receiver, demod, name, freq,
                      audio_device=audio_device, squelch_db=margin_db,
                      audio_factor=audio_factor, force_open=force_open)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="%s - Airspy PMR446 / LPD433 skener" % STATION_NAME)
    ap.add_argument("--list", action="store_true",
                    help="vypsat PMR kanaly a konec")
    ap.add_argument("--list-devices", action="store_true",
                    help="vypsat audio vystupy a konec")
    ap.add_argument("--channel", type=int,
                    help="rovnou poslouchat PMR kanal (1-16)")
    ap.add_argument("--lpd-channel", type=int,
                    help="rovnou poslouchat LPD kanal (1-69)")
    ap.add_argument("--freq", type=float,
                    help="rovnou poslouchat dany kmitocet v MHz")
    ap.add_argument("--lpd", action="store_true",
                    help="pri skenu pridat LPD433 kanaly")
    ap.add_argument("--rate", type=float, default=2.5e6,
                    help="sample rate (default 2.5M)")
    ap.add_argument("--gain", type=float, default=30, help="RF gain dB")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="squelch: dB nad sumovym dnem (default 1.0)")
    ap.add_argument("--audio-gate", type=float, default=3.0,
                    help="audio squelch: nasobek audio sumu (default 3.0)")
    ap.add_argument("--force-open", action="store_true",
                    help="pro test - vzdy otevreny squelch")
    ap.add_argument("--dwell", type=float, default=0.15,
                    help="doba na kanal pri skenu (s)")
    ap.add_argument("--device", default=None,
                    help="audio vystup: index nebo substring jmena")
    ap.add_argument("--no-banner", action="store_true",
                    help="nevypisovat uvodni banner")
    args = ap.parse_args()

    if not args.no_banner:
        print_banner()

    if args.list_devices:
        if sd is None:
            sys.exit("sounddevice neni nainstalovan")
        print("[%s] Audio vystupy:" % STATION_SHORT)
        for idx, d in enumerate(sd.query_devices()):
            if d["max_output_channels"] > 0:
                print("  #%d  %s  (out ch=%d, hostapi=%s)" %
                      (idx, d["name"], d["max_output_channels"],
                       sd.query_hostapis(d["hostapi"])["name"]))
        auto = find_audio_device(args.device)
        print("\n[%s] Automaticky bych vybral: %s"
              % (STATION_SHORT, describe_device(auto)))
        print_footer()
        return

    if args.list:
        print("[%s] PMR446 kanaly:" % STATION_SHORT)
        for i, f in enumerate(PMR446, 1):
            print("  PMR%-3d %.5f MHz" % (i, f / 1e6))
        if args.lpd:
            print("[%s] LPD433 kanaly:" % STATION_SHORT)
            for i, f in enumerate(LPD433, 1):
                print("  LPD%-3d %.5f MHz" % (i, f / 1e6))
        print_footer()
        return

    direct_target = None
    if args.channel is not None:
        if not 1 <= args.channel <= 16:
            sys.exit("PMR kanal je 1-16")
        direct_target = ("PMR%d" % args.channel, PMR446[args.channel - 1])
    elif args.lpd_channel is not None:
        if not 1 <= args.lpd_channel <= 69:
            sys.exit("LPD kanal je 1-69")
        direct_target = ("LPD%d" % args.lpd_channel,
                         LPD433[args.lpd_channel - 1])
    elif args.freq is not None:
        if not 1.0 <= args.freq <= 6000.0:
            sys.exit("Kmitocet mimo rozsah (1-6000 MHz)")
        hz = int(round(args.freq * 1e6))
        direct_target = ("%.5f MHz" % (hz / 1e6), hz)

    channels = make_channels(pmr=True, lpd=args.lpd)

    audio_device = None
    if sd is not None:
        audio_device = find_audio_device(args.device)
        print("[%s] Audio vystup: %s"
              % (STATION_SHORT, describe_device(audio_device)))
    else:
        print("[!] sounddevice neni instalovan - bez prehravani")

    print("[%s] Inicializace Airspy..." % STATION_SHORT)
    rx = AirspyReceiver(sample_rate=args.rate, gain=args.gain)
    demod = NbfmDemod(in_rate=args.rate, out_rate=48000)

    try:
        if direct_target is not None:
            listen_single(rx, demod, direct_target[0], direct_target[1],
                          audio_device=audio_device,
                          squelch_db=args.threshold,
                          audio_factor=args.audio_gate,
                          force_open=args.force_open)
        else:
            scan(rx, demod, channels, dwell=args.dwell,
                 margin_db=args.threshold, audio_device=audio_device,
                 audio_factor=args.audio_gate, force_open=args.force_open)
    finally:
        try:
            rx.sdr.deactivateStream(rx.rx_stream)
            rx.sdr.closeStream(rx.rx_stream)
        except Exception:
            pass
        print_footer()


if __name__ == "__main__":
    main()