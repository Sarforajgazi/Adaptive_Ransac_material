# TartanGround & Open3D Ground Segmentation Evaluation Report

This report summarizes the performance, execution times, and geometric results of our C++ Cython-wrapped RANSAC ground segmentation pipeline across six distinct datasets.

---

## 📊 Summary of Ground Segmentation Results

| Environment | Type | Coordinate System | Original Points | Voxel Size | Downsampled Points | Detected Planes | Ground Points | Ground Z-Height | Normal Alignment | Total Execution Time |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Office Room** | Indoor (Default) | **Z-Up** | 15,678 | $0.02\text{ m}$ | 15,678 | 1 | 11,251 | **$1.37\text{ m}$** | $1.000$ | $0.05\text{ s}$ |
| **Supermarket** | Large Indoor | **Z-Down** | 7,793,836 | $0.30\text{ m}$ | 157,802 | 7 | 16,520 | **$2.10\text{ m}$** | $1.000$ | $5.49\text{ s}$ |
| **SeasonalForestAutumn** | Rugged Forest | **Z-Down** | 32,286,576 | $1.50\text{ m}$ | 159,790 | 2 | 13,208 | **$45.31\text{ m}$** | $0.999$ | $23.76\text{ s}$ |
| **OldScandinavia** | Mountain/Forest | **Z-Down** | 123,136,181 | $1.20\text{ m}$ | 645,956 | 2 | 17,742 | **$13.09\text{ m}$** | $0.999$ | $96.56\text{ s}$ |
| **SeasonalForestSpring** | Rugged Forest | **Z-Down** | 31,701,720 | $1.50\text{ m}$ | 129,745 | 2 | 24,003 | **$38.57\text{ m}$** | $1.000$ | $20.93\text{ s}$ |
| **Sewerage** | Tight Tunnels | **Z-Down** | 7,923,456 | $0.30\text{ m}$ | 102,249 | 4 | 3,760 | **$6.54\text{ m}$** | $1.000$ | $5.56\text{ s}$ |

---

## 🔍 Detailed Log per Environment

### 1. Default Office Room Dataset
* **Description:** Small indoor office scan from Open3D datasets.
* **Logs & Metrics:**
  * **Loading Time:** Fast (local cache)
  * **Plane Detection Time:** $0.050\text{ seconds}$
  * **Ground Z Elevation:** $1.368\text{ m}$ (Z-Up convention: ground plane corresponds to the **lowest** vertical coordinates).
  * **Ground Points:** $11,251\text{ points}$ ($\sim 71.7\%$ of the downsampled points).

---

### 2. Supermarket Dataset
* **Description:** Global dense point cloud of a commercial retail environment with shelves, registers, and walkways.
* **Logs & Metrics:**
  * **Loading Time:** $3.39\text{ seconds}$
  * **Bounding Box Extents:** $X=301.96\text{ m}$, $Y=256.07\text{ m}$, $Z=22.40\text{ m}$
  * **Voxel Downsampling Time:** $1.31\text{ seconds}$
  * **Plane Detection Time:** $0.79\text{ seconds}$
  * **Ground Z Elevation:** $2.10\text{ m}$ (Z-Down/NED convention: ground plane corresponds to the **highest** vertical coordinates).
  * **Ground Points:** $16,520\text{ points}$ ($\sim 10.5\%$ of downsampled points).
  * **Saved Outputs:**
    * Original PLY: `tartanair_data/Supermarket/Supermarket_rgb_original.ply`
    * Segmented PLY: `tartanair_data/Supermarket/Supermarket_rgb_segmented.ply`

---

### 3. SeasonalForestAutumn Dataset
* **Description:** Dense outdoor forest environment containing trees, leaves, and uneven terrain.
* **Logs & Metrics:**
  * **Loading Time:** $15.38\text{ seconds}$
  * **Bounding Box Extents:** $X=447.55\text{ m}$, $Y=545.33\text{ m}$, $Z=62.17\text{ m}$
  * **Voxel Downsampling Time:** $7.35\text{ seconds}$
  * **Plane Detection Time:** $1.03\text{ seconds}$
  * **Ground Z Elevation:** $45.31\text{ m}$ (Z-Down/NED convention)
  * **Ground Points:** $13,208\text{ points}$ ($\sim 8.3\%$ of downsampled points).
  * **Saved Outputs:**
    * Original PLY: `tartanair_data/SeasonalForestAutumn/SeasonalForestAutumn_rgb_original.ply`
    * Segmented PLY: `tartanair_data/SeasonalForestAutumn/SeasonalForestAutumn_rgb_segmented.ply`

---

### 4. OldScandinavia Dataset
* **Description:** Massive rugged terrain environment representing old Scandinavian mountains and vegetation.
* **Logs & Metrics:**
  * **Loading Time:** $65.76\text{ seconds}$
  * **Bounding Box Extents:** $X=595.83\text{ m}$, $Y=609.46\text{ m}$, $Z=127.13\text{ m}$
  * **Voxel Downsampling Time:** $26.75\text{ seconds}$
  * **Plane Detection Time:** $4.05\text{ seconds}$ (with fixed RANSAC `min_support` cap at 5,000 points)
  * **Ground Z Elevation:** $13.09\text{ m}$ (Z-Down/NED convention)
  * **Ground Points:** $17,742\text{ points}$ ($\sim 2.7\%$ of downsampled points).
  * **Saved Outputs:**
    * Original PLY: `tartanair_data/OldScandinavia/OldScandinavia_rgb_original.ply`
    * Segmented PLY: `tartanair_data/OldScandinavia/OldScandinavia_rgb_segmented.ply`

---

### 5. SeasonalForestSpring Dataset
* **Description:** Lush forest environment with dense vegetation and dirt paths during the spring season.
* **Logs & Metrics:**
  * **Loading Time:** $15.65\text{ seconds}$
  * **Bounding Box Extents:** $X=466.02\text{ m}$, $Y=408.68\text{ m}$, $Z=53.18\text{ m}$
  * **Voxel Downsampling Time:** $4.78\text{ seconds}$
  * **Plane Detection Time:** $0.60\text{ seconds}$
  * **Ground Z Elevation:** $38.57\text{ m}$ (Z-Down/NED convention)
  * **Ground Points:** $24,003\text{ points}$ ($\sim 18.5\%$ of downsampled points).
  * **Saved Outputs:**
    * Original PLY: `tartanair_data/SeasonalForestSpring/SeasonalForestSpring_rgb_original.ply`
    * Segmented PLY: `tartanair_data/SeasonalForestSpring/SeasonalForestSpring_rgb_segmented.ply`

---

### 6. Sewerage Dataset
* **Description:** Tight underground sewer channels and pipelines.
* **Logs & Metrics:**
  * **Loading Time:** $3.74\text{ seconds}$
  * **Bounding Box Extents:** $X=104.78\text{ m}$, $Y=75.96\text{ m}$, $Z=28.86\text{ m}$
  * **Voxel Downsampling Time:** $1.40\text{ seconds}$
  * **Plane Detection Time:** $0.42\text{ seconds}$
  * **Ground Z Elevation:** $6.54\text{ m}$ (Z-Down/NED convention)
  * **Ground Points:** $3,760\text{ points}$ ($\sim 3.7\%$ of downsampled points).
  * **Saved Outputs:**
    * Original PLY: `tartanair_data/Sewerage/Sewerage_rgb_original.ply`
    * Segmented PLY: `tartanair_data/Sewerage/Sewerage_rgb_segmented.ply`

---

## 🛠️ Verification Commands Quick-Sheet

You can re-run and visualize any of these environments by executing these commands from the `schnabel_cython` directory:

1. **Office Room:**
   ```bash
   python ground_segmentation.py
   ```
2. **Supermarket:**
   ```bash
   python tartan_ground_segmentation.py tartanair_data/Supermarket/Supermarket_rgb.pcd z_down 0.3
   ```
3. **Autumn Forest:**
   ```bash
   python tartan_ground_segmentation.py tartanair_data/SeasonalForestAutumn/SeasonalForestAutumn_rgb.pcd z_down 1.5
   ```
4. **Old Scandinavia:**
   ```bash
   python tartan_ground_segmentation.py tartanair_data/OldScandinavia/OldScandinavia_rgb.pcd z_down 1.2
   ```
5. **Spring Forest:**
   ```bash
   python tartan_ground_segmentation.py tartanair_data/SeasonalForestSpring/SeasonalForestSpring_rgb.pcd z_down 1.5
   ```
6. **Sewerage Tunnels:**
   ```bash
   python tartan_ground_segmentation.py tartanair_data/Sewerage/Sewerage_rgb.pcd z_down 0.3
   ```
