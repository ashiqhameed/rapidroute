import holidays
import numpy as np
import pandas as pd

from src.config import CFG, path

USECOLS = [
    "incident_id", "incident_datetime", "initial_call_type",
    "initial_severity_level_code", "final_severity_level_code",
    "valid_incident_rspns_time_indc", "incident_response_seconds_qy",
    "incident_travel_tm_seconds_qy", "dispatch_response_seconds_qy",
    "held_indicator", "incident_dispatch_area", "zipcode",
    "first_assignment_datetime", "incident_close_datetime",
]
MIN_ZIP_CALLS = 500
T_MIN, T_MAX = 30, 3600


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def load_zcta():
    z = pd.read_csv(path("zcta_raw"), sep="\t", dtype={"GEOID": int})
    z.columns = z.columns.str.strip()
    return z.rename(columns={"GEOID": "zip", "INTPTLAT": "lat", "INTPTLONG": "lon", "ALAND_SQMI": "area_sqmi"})[
        ["zip", "lat", "lon", "area_sqmi"]
    ]


def build_zones(df, zcta):
    counts = df["zip_raw"].value_counts().rename("calls").rename_axis("zip").reset_index()
    counts = counts.merge(zcta, on="zip", how="left")
    keep = counts[(counts.calls >= MIN_ZIP_CALLS) & counts.lat.notna()].copy()
    small = counts[~counts.zip.isin(keep.zip)].copy()

    mapping = dict(zip(keep.zip, keep.zip))
    for _, r in small.iterrows():
        if pd.isna(r.lat):
            continue
        d = haversine_km(r.lat, r.lon, keep.lat.values, keep.lon.values)
        mapping[r.zip] = int(keep.zip.values[d.argmin()])

    zones = keep[["zip", "lat", "lon", "area_sqmi"]].reset_index(drop=True)
    zones["zone_id"] = np.arange(len(zones))
    return zones, mapping


def active_count(starts, ends, t):
    n = np.searchsorted(np.sort(starts), t, side="left") - np.searchsorted(np.sort(ends), t, side="right")
    return np.maximum(n, 0)


def add_system_load(df, raw):
    opened = pd.to_datetime(raw.incident_datetime)
    closed = pd.to_datetime(raw.incident_close_datetime).fillna(opened + pd.Timedelta(hours=1))
    assigned = pd.to_datetime(raw.first_assignment_datetime).fillna(closed)
    o, c, a = (x.values.astype("datetime64[ns]") for x in (opened, closed, assigned))
    t = df.ts.values.astype("datetime64[ns]")
    df["open_city"] = active_count(o, c, t)
    df["waiting_city"] = active_count(o, a, t)
    df["open_area"] = 0
    area = raw.incident_dispatch_area.values
    for ar, idx in df.groupby("incident_dispatch_area").groups.items():
        m = area == ar
        df.loc[idx, "open_area"] = active_count(o[m], c[m], df.loc[idx, "ts"].values.astype("datetime64[ns]"))
    return df


def time_band(h):
    if h < 6:
        return "night"
    if h < 10:
        return "am_peak"
    if h < 16:
        return "midday"
    if h < 20:
        return "pm_peak"
    return "evening"


def severity_bucket(s):
    if s <= 3:
        return "critical"
    if s <= 5:
        return "urgent"
    return "non_urgent"


def clean_seconds(s, valid):
    s = pd.to_numeric(s, errors="coerce")
    return s.where(valid & s.between(T_MIN, T_MAX))


def main():
    raw = pd.read_csv(path("ems_raw"), usecols=USECOLS, low_memory=False)
    n0 = len(raw)
    df = raw.dropna(subset=["zipcode", "initial_severity_level_code"]).copy()
    df["zip_raw"] = df["zipcode"].astype(int)
    df = df[df.zip_raw.between(10001, 10282)]

    zcta = load_zcta()
    zones, mapping = build_zones(df, zcta)
    df["zip"] = df.zip_raw.map(mapping)
    df = df.dropna(subset=["zip"])
    df["zip"] = df.zip.astype(int)
    df = df.merge(zones, on="zip", how="left")

    t = pd.to_datetime(df.incident_datetime)
    df["ts"] = t
    df["date"] = t.dt.normalize()
    df["hour"] = t.dt.hour
    df["dow"] = t.dt.dayofweek
    df["month"] = t.dt.month
    df["how"] = df.dow * 24 + df.hour
    df["is_weekend"] = (df.dow >= 5).astype(int)
    hol = holidays.country_holidays("US", subdiv="NY", years=CFG["year"])
    df["is_holiday"] = df.date.dt.date.map(lambda d: int(d in hol))
    df["hour_sin"] = np.sin(2 * np.pi * df.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df.dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.dow / 7)
    df["time_band"] = df.hour.map(time_band)

    df["severity"] = df.initial_severity_level_code.astype(int)
    df["sev_bucket"] = df.severity.map(severity_bucket)
    df["is_critical"] = (df.sev_bucket == "critical").astype(int)

    valid = df.valid_incident_rspns_time_indc.eq("Y")
    df["response_s"] = clean_seconds(df.incident_response_seconds_qy, valid)
    df["travel_s"] = clean_seconds(df.incident_travel_tm_seconds_qy, valid)
    df["dispatch_s"] = pd.to_numeric(df.dispatch_response_seconds_qy, errors="coerce")
    df["late"] = (df.response_s > CFG["response_target_seconds"]).astype("Int8").where(df.response_s.notna())

    df["split"] = np.where(df.month.isin(CFG["test_months"]), "test", "train")
    df = add_system_load(df, raw)

    cols = [
        "incident_id", "ts", "date", "hour", "dow", "month", "how", "is_weekend", "is_holiday",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "time_band", "initial_call_type",
        "severity", "sev_bucket", "is_critical", "zip", "zip_raw", "zone_id", "lat", "lon",
        "area_sqmi", "incident_dispatch_area", "held_indicator", "open_city", "open_area",
        "waiting_city", "dispatch_s", "travel_s", "response_s", "late", "split",
    ]
    df = df[cols].sort_values("ts").reset_index(drop=True)

    out = path("processed")
    df.to_parquet(out / "incidents.parquet", index=False)
    zones.to_csv(out / "zones.csv", index=False)
    pd.Series(mapping, name="zone_zip").rename_axis("zip_raw").to_csv(out / "zip_to_zone.csv")

    summary = {
        "raw_rows": n0,
        "clean_rows": len(df),
        "zones": len(zones),
        "valid_response_rows": int(df.response_s.notna().sum()),
        "train_rows": int((df.split == "train").sum()),
        "test_rows": int((df.split == "test").sum()),
        "median_response_min": round(df.response_s.median() / 60, 2),
        "median_travel_min": round(df.travel_s.median() / 60, 2),
        "late_rate": round(float(df.late.mean()), 4),
        "critical_share": round(float(df.is_critical.mean()), 4),
    }
    pd.Series(summary).to_json(out / "preprocess_summary.json", indent=2)
    for k, v in summary.items():
        print(f"{k:>22}: {v}")


if __name__ == "__main__":
    main()
