import spiceypy as spice
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.image as mpimg


def mask_local_time(
    lon_grid_deg,
    lon_sub_deg,
    target_time_hours,
    band_width_hours=0.25,
    lon_range="[-180,180)"
):
    """
    Compute a mask for locations whose local solar time is within a given band.
    ACE 20260629
    
    Parameters
    ----------
    lon_grid_deg : ndarray
        2D array of longitudes (degrees) for each pixel on the map.
        Can be in any continuous 360° range (e.g. [-180,180), [0,360), [-360,0)).
    lon_sub_deg : float
        Subsolar longitude (degrees), as returned by SPICE (after any
        desired mapping, e.g. to [-360,0) to match lon_grid_deg).
    target_time_hours : float
        Target local solar time in hours, 0–24. For example:
        - 12.0 = local noon
        - 15.5 = 15:30 (3:30 pm)
    band_width_hours : float, optional
        Half-width of the time band in hours (default 0.25 = ±15 minutes).
        The mask will be True where local time is in:
            [target_time_hours - band_width_hours,
             target_time_hours + band_width_hours]
        with wrap-around at 0/24.
    lon_range : {"[-180,180)", "[0,360)", "[-360,0)"}, optional
        Indicates how lon_grid_deg and lon_sub_deg are laid out.
        This is only used to map them consistently into a 0–360 range for
        local time computation. The mask returned has the same shape
        (and indexing) as lon_grid_deg.

    Returns
    -------
    mask : ndarray (bool)
        Boolean array, same shape as lon_grid_deg, True where local solar
        time is within the specified band.
    """

    Normalize longitudes to a common [0, 360) system for time calculations
    if lon_range == "[-180,180)":
        # Example value: -170 → 190,  170 → 170
        lon_grid_360 = (lon_grid_deg + 360.0) % 360.0
        lon_sub_360 = (lon_sub_deg + 360.0) % 360.0
    elif lon_range == "[0,360)":
        lon_grid_360 = lon_grid_deg % 360.0
        lon_sub_360 = lon_sub_deg % 360.0
    elif lon_range == "[-360,0)":
        # Example value: -360→0, -270→90, -10→350
        lon_grid_360 = (lon_grid_deg + 360.0) % 360.0
        lon_sub_360 = (lon_sub_deg + 360.0) % 360.0
    else:
        raise ValueError(f"Unsupported lon_range: {lon_range}")

    # 2. Longitude difference (hour angle) in degrees, normalized to [-180, 180)
    delta_lon = lon_grid_360 - lon_sub_360
    delta_lon = (delta_lon + 180.0) % 360.0 - 180.0  # now in [-180, 180)

    # 3. Convert longitude difference to local solar time (hours)
    #
    # Convention: LST = 12h + delta_lon / 15°/hour
    # delta_lon > 0 → local afternoon
    # delta_lon < 0 → local morning
    lst = 12.0 + delta_lon / 15.0  # may be outside 0–24
    lst = lst % 24.0              # bring into [0, 24)

    # 4. Build band around target_time_hours, with wrap-around at 0/24
    t0 = (target_time_hours - band_width_hours) % 24.0
    t1 = (target_time_hours + band_width_hours) % 24.0

    if t0 <= t1:
        # Simple case: band does not cross midnight
        mask = (lst >= t0) & (lst <= t1)
    else:
        # Band wraps around midnight, e.g. target 23.8h ±0.5h:
        # [23.3, 24) ∪ [0, 0.3]
        mask = (lst >= t0) | (lst <= t1)
    
    return mask,(lon_grid*mask).flatten().min(),(lon_grid*mask).flatten().max()


def generate_map(output_path: Path) -> None:

  spice.furnsh("naif0012.tls")      # leapseconds
  spice.furnsh("de432s.bsp")         # solar system ephemeris
  spice.furnsh("sat375.bsp")
  spice.furnsh("pck00010.tpc")      # planetary constants & radii

  datetoplot=dt.datetime.now().isoformat()
  ephtime=spice.utc2et(datetoplot)
  subslr_method = "Near point: ellipsoid"   # or "Intercept"
  target = "TITAN"
  fixref = "IAU_TITAN"
  abcorr = "LT+S"
  obsrvr = "SUN"
  
  spoint, trgepc, srfvec = spice.subslr(subslr_method,target,ephtime,fixref,abcorr,obsrvr)
  
  # Convert Cartesian to lat/lon in radians
  radius, lon, lat = spice.reclat(spoint)
  # Convert to degrees
  lon_deg = spice.dpr() * lon 
  lat_deg = spice.dpr() * lat
  
  lon_neg360 = (lon_deg - 360.0) % 360.0
  if lon_neg360 >= 0:
      lon_neg360 -= 360.0


  # Sun position relative to Titan in IAU_TITAN frame
  # State: position + velocity, we only need position.
  state_sun_titan, _ = spice.spkezr(
      "SUN",   # target
      ephtime,
      "IAU_TITAN",
      abcorr,
      "TITAN" # observer
  )

  sun_vec = state_sun_titan[:3]
  # Unit vector from Titan center toward Sun:
  sun_dir = sun_vec / spice.vnorm(sun_vec)

  n_lon = basemap.shape[0] # example
  n_lat = basemap.shape[1]
  
  lons = np.linspace(-360, 0, n_lon)  # deg
  lats = np.linspace(-90, 90, n_lat)    # deg
  
  lon_grid, lat_grid = np.meshgrid(lons, lats)

  radii = spice.bodvrd("TITAN", "RADII", 3)[1]
  r_titan = (radii[0] + radii[1] + radii[2]) / 3.0
  
  # Convert degrees to radians
  lon_rad = np.deg2rad(lon_grid)
  lat_rad = np.deg2rad(lat_grid)
  
  # Planetocentric -> Cartesian (assuming spherical Titan)
  x = r_titan * np.cos(lat_rad) * np.cos(lon_rad)
  y = r_titan * np.cos(lat_rad) * np.sin(lon_rad)
  z = r_titan * np.sin(lat_rad)
  
  # Surface normal = radial vector direction (since we assumed spherical)
  normal_x = x / r_titan
  normal_y = y / r_titan
  normal_z = z / r_titan
  
  # Broadcast sun_dir components
  sx, sy, sz = sun_dir
  
  dot = normal_x * sx + normal_y * sy + normal_z * sz
  
  # sun is above horizon if dot > 0
  day_mask = dot > 0
  night_mask = ~day_mask

  dflylocaltime=12+(dflylon-lon_neg360)/15
  dflylocaltimestr=str(np.floor(dflylocaltime))[:-2]+'h '+str((dflylocaltime-np.floor(dflylocaltime))*60)[0:2]+'m'
  if dflylocaltime <10:
    dflylocaltimestr='0'+str(np.floor(dflylocaltime))[:-2]+'h '+str((dflylocaltime-np.floor(dflylocaltime))*60)[0:2]+'m'



  fig, ax = plt.subplots(figsize=(10, 5))
  
  ax.imshow( basemap, extent=[-360, 0, -90, 90],  # lon_min, lon_max, lat_min, lat_max
      origin="upper",cmap='Greys_r')
  
  # Create a semi-transparent black overlay where night_mask is True
  # Convert boolean night_mask to alpha channel
  alpha = np.zeros_like(night_mask, dtype=float)
  alpha[night_mask] = 0.6  # 0.6 opacity for night
  
  # We can plot a black image with alpha varying
  shadow = np.zeros((n_lat, n_lon, 4))  # RGBA
  shadow[..., 3] = alpha  # only alpha channel
  
  ax.imshow(shadow,extent=[-360, 0, -90, 90],origin="upper")
  
  
  # mark subsolar point
  ax.scatter(lon_deg, lat_deg, color="#FDB71A", s=50, marker='*', zorder=5)
  ax.scatter(dflylon,dflylat,s=10,color='#AD293D')
  ax.set_xlabel("Longitude (deg)")
  ax.set_ylabel("Latitude (deg)")
  ax.set_title(f"Titan Day/Night at {datetoplot} \n Dfly Landing Ellipse LST {dflylocaltimestr}")
  
  plt.tight_layout()
  fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
  plt.close(fig)


def main():
  repo_root = Path(__file__).resolve().parents[1]
  output_path = repo_root / "assets" / "img" / "current_titan_map.png"
  generate_titan_map(output_path)


if __name__ == "__main__":
    main()
