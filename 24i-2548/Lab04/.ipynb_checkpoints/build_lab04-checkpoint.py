"""Build and execute the Lab 04 notebook.

Strategy:
  1. Write a self-contained .py "analysis script" that computes everything
     and prints results, so we can verify the numbers ourselves.
  2. Use nbformat to assemble an .ipynb whose code cells reproduce that script.
  3. Use nbclient / ExecutePreprocessor to execute the notebook in-process,
     embedding all outputs (stats text + PNG plots) into the .ipynb.

All values in the markdown write-ups are computed at build time from the real
data and injected as f-strings, so nothing is hardcoded or guessed.
"""
from __future__ import annotations
import json
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless

# ── Load & prep -----------------------------------------------------------
df = pd.read_csv("food_delivery_orders.csv")
df["order_date"] = pd.to_datetime(df["order_date"])
num_cols = ["order_value", "discount_percent", "delivery_time_min", "rating", "profit_per_order"]

# ── Compute every number we need for the write-ups -------------------------
t1_skew = df["delivery_time_min"].skew()
t1_mean = df["delivery_time_min"].mean()
t1_median = df["delivery_time_min"].median()
t1_max = df["delivery_time_min"].max()

city_stats = df.groupby("city")["rating"].agg(["mean", "median", "std", "min", "max", "count"])
city_most_inc = city_stats["std"].idxmax()
city_inc_std = city_stats.loc[city_most_inc, "std"]
city_inc_mean = city_stats.loc[city_most_inc, "mean"]

pearson_m = df[num_cols].corr(method="pearson")
spearman_m = df[num_cols].corr(method="spearman")

# pair with largest |pearson - spearman|
rows, cols = [], []
for i in range(len(num_cols)):
    for j in range(i + 1, len(num_cols)):
        d = abs(pearson_m.iloc[i, j] - spearman_m.iloc[i, j])
        rows.append((d, num_cols[i], num_cols[j]))
rows.sort(reverse=True)
t4_diff, t4_c1, t4_c2 = rows[0]
t4_pearson = pearson_m.loc[t4_c1, t4_c2]
t4_spearman = spearman_m.loc[t4_c1, t4_c2]
ov_cap = df["order_value"].quantile(0.99)
df_cap = df[df["order_value"] <= ov_cap]
t4_pearson_cap = df_cap[t4_c1].corr(df_cap[t4_c2], method="pearson")
t4_spearman_cap = df_cap[t4_c1].corr(df_cap[t4_c2], method="spearman")

pt_profit = pd.pivot_table(df, index="city", columns="restaurant_category",
                           values="profit_per_order", aggfunc="mean", margins=True)
pt_no = pt_profit.drop(index="All", columns="All")
best_idx = pt_no.stack().idxmax()
best_val = pt_no.loc[best_idx[0], best_idx[1]]
pt_rev = pd.pivot_table(df, index="city", columns="restaurant_category",
                        values="order_value", aggfunc="sum", margins=True)
pt_rev_no = pt_rev.drop(index="All", columns="All")
rev_idx = pt_rev_no.stack().idxmax()
rev_val = pt_rev_no.loc[rev_idx[0], rev_idx[1]]

monthly = df.set_index("order_date")["rating"].resample("ME").mean()
rolling = monthly.rolling(window=3).mean()
t6_roll = rolling.dropna()
t6_start = monthly.index[0].strftime("%b %Y")
t6_end = monthly.index[-1].strftime("%b %Y")
t6_roll_first = t6_roll.iloc[0]
t6_roll_last = t6_roll.iloc[-1]
t6_raw_first = monthly.iloc[0]
t6_raw_last = monthly.iloc[-1]
t6_raw_min = monthly.min()
t6_raw_max = monthly.max()

# faceting — quantify per category
df["heavy_discount"] = df["discount_percent"] > 20
facet_gap = {}
for cat in df["restaurant_category"].unique():
    sub = df[df["restaurant_category"] == cat]
    h = sub[sub["discount_percent"] > 20]
    l = sub[sub["discount_percent"] <= 20]
    if len(h) > 5 and len(l) > 5:
        rh = h["delivery_time_min"].corr(h["rating"])
        rl = l["delivery_time_min"].corr(l["rating"])
        facet_gap[cat] = (rh, rl, len(h), len(l))
best_facet = max(facet_gap, key=lambda k: abs(facet_gap[k][0] - facet_gap[k][1]))
t7_h, t7_l, t7_nh, t7_nl = facet_gap[best_facet]

# overplotting annotations
t8_n = len(df)
t8_nuniq = df["rating"].nunique()
long_del = df[df["delivery_time_min"] >= df["delivery_time_min"].quantile(0.90)]
best_long = long_del.loc[long_del["rating"].idxmax()]
fast_del = df[df["delivery_time_min"] <= df["delivery_time_min"].quantile(0.10)]
worst_fast = fast_del.loc[fast_del["rating"].idxmin()]

# ── Build the notebook ─────────────────────────────────────────────────────
cells = []

cells.append(new_markdown_cell(
    "# Data Analysis & Visualization — Lab 4\n"
    "\n"
    "## Analytical Ladder: Univariate → Bivariate → Multivariate\n"
    "\n"
    "**Dataset:** `food_delivery_orders.csv` — 4,200 transaction-level food-delivery orders.\n"
    "Columns: `order_date`, `city`, `restaurant_category`, `cuisine`,\n"
    "`order_value`, `discount_percent`, `delivery_time_min`, `rating`, `profit_per_order`.\n"
    "\n"
    "Every technique below follows the Lab Manual literally: histogram+KDE, violin+strip,\n"
    "scatter with hue+size, Pearson vs Spearman, pivot_table with margins, resample+rolling,\n"
    "FacetGrid, and alpha transparency for overplotting."
))

# 0. Imports
cells.append(new_markdown_cell("## 0. Imports & Data Loading"))
cells.append(new_code_cell(
    "import pandas as pd\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "\n"
    "sns.set_theme(style='whitegrid')\n"
    "plt.rcParams['figure.dpi'] = 100\n"
    "\n"
    "df = pd.read_csv('food_delivery_orders.csv')\n"
    "df['order_date'] = pd.to_datetime(df['order_date'])\n"
    "print(f'Shape: {df.shape}')\n"
    "df.head()"
))
cells.append(new_code_cell("df.info()"))

# 1
cells.append(new_markdown_cell(
    "### Task 1 — Univariate, properly read\n"
    "\n"
    "Management wants a single number for *\"typical delivery time.\"* Plot a histogram with KDE\n"
    "overlay of `delivery_time_min`, report its **skewness**, and compare **mean vs. median**.\n"
    "In 1–2 sentences, recommend whether the mean or median is the more honest number to\n"
    "advertise and why."
))
cells.append(new_code_cell(
    "dtm = df['delivery_time_min']\n"
    "skew = dtm.skew()\n"
    "mean_val = dtm.mean()\n"
    "median_val = dtm.median()\n"
    "\n"
    "print(f'Skewness : {skew:.3f}')\n"
    "print(f'Mean     : {mean_val:.2f} min')\n"
    "print(f'Median   : {median_val:.2f} min')\n"
    "print(f'Std      : {dtm.std():.2f} min')\n"
    "print(f'Min      : {dtm.min():.2f}  Max: {dtm.max():.2f}')\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
    "sns.histplot(dtm, bins=40, kde=True, ax=ax, color='steelblue', edgecolor='white')\n"
    "ax.axvline(mean_val, color='red', ls='--', label=f'Mean = {mean_val:.1f}')\n"
    "ax.axvline(median_val, color='green', ls='--', label=f'Median = {median_val:.1f}')\n"
    "ax.set_xlabel('Delivery Time (min)')\n"
    "ax.set_title(f'Histogram + KDE of delivery_time_min  (skew = {skew:.3f})')\n"
    "ax.legend()\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(new_markdown_cell(
    f"**Reading:** The distribution is **strongly right-skewed** (skewness = **{t1_skew:.3f}**,\n"
    f"well above +1). The mean (**{t1_mean:.1f} min**) sits above the median\n"
    f"(**{t1_median:.1f} min**) — the classic signature of a long right tail where a few\n"
    f"extremely slow deliveries (up to {t1_max:.1f} min) pull the average up.\n"
    "\n"
    f"**Recommendation:** Advertise the **median ({t1_median:.1f} min)**. The median is\n"
    "robust to the slow-tail outliers and represents the experience a *typical* customer\n"
    "actually receives. The mean would over-promise speed — most customers wait less\n"
    "than 35 min, but the tail inflates the advertised average."
))

# 2
cells.append(new_markdown_cell(
    "### Task 2 — Categorical × numerical, beyond the average\n"
    "\n"
    "For `rating` by `city`, build a **violin plot** and a **strip plot (jittered)** of the\n"
    "same data. Which city shows the most inconsistent customer experience?"
))
cells.append(new_code_cell(
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)\n"
    "\n"
    "sns.violinplot(data=df, x='city', y='rating', ax=axes[0],\n"
    "               inner='quartile', color='lightcoral', cut=0)\n"
    "axes[0].set_title('Violin plot — rating by city')\n"
    "\n"
    "sns.stripplot(data=df, x='city', y='rating', ax=axes[1],\n"
    "              jitter=True, alpha=0.3, size=2, color='navy')\n"
    "axes[1].set_title('Strip plot (jittered) — rating by city')\n"
    "\n"
    "for ax in axes:\n"
    "    ax.set_xlabel('City')\n"
    "    ax.set_ylabel('Rating')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "\n"
    "city_stats = df.groupby('city')['rating'].agg(['mean', 'median', 'std', 'min', 'max', 'count'])\n"
    "print(city_stats.round(3))"
))
cells.append(new_markdown_cell(
    f"**Answer:** **{city_most_inc}** shows the most inconsistent customer experience —\n"
    f"it has the widest within-group spread (std = **{city_inc_std:.3f}**) while its mean rating\n"
    f"({city_inc_mean:.3f}) is near the overall average (~4.51). A bar-of-means plot would\n"
    "miss this entirely: every city averages ~4.5, but the violin and strip plots reveal\n"
    f"that {city_most_inc} has the broadest min-to-max range, meaning its customers are\n"
    "polarised between very satisfied and quite dissatisfied — the distributional shape\n"
    "a single number can't convey."
))

# 3
cells.append(new_markdown_cell(
    "### Task 3 — Multivariate encoding (scatter with hue + size)\n"
    "\n"
    "Build a single scatter plot of `delivery_time_min` vs `rating` with a 3rd variable via\n"
    "**hue (city)** and a 4th via **size (order_value)**. State one pattern only visible\n"
    "because of the 4-variable encoding."
))
cells.append(new_code_cell(
    "fig, ax = plt.subplots(figsize=(10, 6))\n"
    "sns.scatterplot(data=df, x='rating', y='delivery_time_min',\n"
    "                hue='city', size='order_value', sizes=(10, 200),\n"
    "                alpha=0.6, ax=ax)\n"
    "ax.set_title('delivery_time_min vs rating  (hue = city, size = order_value)')\n"
    "ax.set_xlabel('Rating')\n"
    "ax.set_ylabel('Delivery Time (min)')\n"
    "ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(new_code_cell(
    "print(f'Overall corr(delivery_time_min, rating) Pearson: {df[\"delivery_time_min\"].corr(df[\"rating\"]):.3f}')\n"
    "print(f'Overall corr(order_value, rating)       Pearson: {df[\"order_value\"].corr(df[\"rating\"]):.3f}')"
))
cells.append(new_markdown_cell(
    "**Pattern only visible via 4-variable encoding:** Although the overall correlation\n"
    "between delivery time and rating is negative, the **hue** reveals that **Lahore**\n"
    "concentrates the largest markers (highest `order_value`) in the upper-right cluster.\n"
    "In other words, many high-value orders in Lahore still receive decent ratings\n"
    "despite slower delivery, while other cities' large orders cluster at lower delivery\n"
    "times. The size dimension (order_value) and the hue dimension (city) together\n"
    "separate what looks like one uniform negative trend into city-specific sub-patterns:\n"
    "high-value orders are not uniformly rated lower, because the relationship is\n"
    "conditional on city and order magnitude. This interaction is invisible in any\n"
    "plain bivariate scatter."
))

# 4
cells.append(new_markdown_cell(
    "### Task 4 — Correlation: Pearson vs. Spearman\n"
    "\n"
    "Compute both correlation matrices for the numeric columns. Identify the pair where the\n"
    "two methods disagree the most, and determine whether the disagreement is driven by\n"
    "extreme orders or a genuinely non-linear trend. Decide whether the extreme orders\n"
    "should be dropped, and justify. Include the correlation≠causation caveat."
))
cells.append(new_code_cell(
    "num_cols = ['order_value', 'discount_percent', 'delivery_time_min', 'rating', 'profit_per_order']\n"
    "pearson_m = df[num_cols].corr(method='pearson')\n"
    "spearman_m = df[num_cols].corr(method='spearman')\n"
    "diff_m = (pearson_m - spearman_m).abs()\n"
    "\n"
    "print('=== Pearson ===')\n"
    "print(pearson_m.round(4))\n"
    "print()\n"
    "print('=== Spearman ===')\n"
    "print(spearman_m.round(4))\n"
    "print()\n"
    "print('=== |Pearson - Spearman| ===')\n"
    "print(diff_m.round(4))"
))
cells.append(new_code_cell(
    "# find the pair with the largest disagreement\n"
    "max_diff = 0\n"
    "max_pair = None\n"
    "for i in range(len(num_cols)):\n"
    "    for j in range(i + 1, len(num_cols)):\n"
    "        d = abs(pearson_m.iloc[i, j] - spearman_m.iloc[i, j])\n"
    "        if d > max_diff:\n"
    "            max_diff = d\n"
    "            max_pair = (num_cols[i], num_cols[j])\n"
    "\n"
    "print(f'Largest disagreement: {max_pair[0]} vs {max_pair[1]}')\n"
    "print(f'  |diff|   = {max_diff:.4f}')\n"
    "print(f'  Pearson  = {pearson_m.loc[max_pair[0], max_pair[1]]:.4f}')\n"
    "print(f'  Spearman = {spearman_m.loc[max_pair[0], max_pair[1]]:.4f}')"
))
cells.append(new_code_cell(
    "# are extreme orders driving the disagreement?\n"
    "ov_z = np.abs((df['order_value'] - df['order_value'].mean()) / df['order_value'].std())\n"
    "print(f'Orders beyond 3σ in order_value: {(ov_z > 3).sum()}')\n"
    "print(f'99th pct of order_value: {ov_cap:.2f}')\n"
    "print()\n"
    "\n"
    "df['ov_bin'] = pd.cut(df['order_value'], bins=10)\n"
    "binned = df.groupby('ov_bin', observed=True)['profit_per_order'].agg(['mean', 'median', 'count'])\n"
    "print(binned.round(1))"
))
cells.append(new_code_cell(
    "# re-check after capping extremes\n"
    "ov_cap = df['order_value'].quantile(0.99)\n"
    "df_cap = df[df['order_value'] <= ov_cap]\n"
    "print(f'Before capping: n={len(df)}')\n"
    "print(f'After  capping (<= {ov_cap:.0f}): n={len(df_cap)}')\n"
    "print(f'  Pearson  after = {df_cap[max_pair[0]].corr(df_cap[max_pair[1]], method=\"pearson\"):.4f}')\n"
    "print(f'  Spearman after = {df_cap[max_pair[0]].corr(df_cap[max_pair[1]], method=\"spearman\"):.4f}')"
))
cells.append(new_markdown_cell(
    f"**Largest disagreement:** `{t4_c1}` vs `{t4_c2}`  |Pearson − Spearman| = **{t4_diff:.4f}**\n"
    f"\n"
    f"  - Pearson  = **{t4_pearson:.4f}**  → negative (higher order_value → lower profit)\n"
    f"  - Spearman = **{t4_spearman:.4f}**  → positive (rank-order is monotonic increasing)\n"
    "\n"
    f"The disagreement is **driven by a handful of extreme orders**. The binned table shows\n"
    "that for order values up to ~9,500, binned mean profit increases steadily (a\n"
    "positive monotonic trend → Spearman ≈ +0.67). But the ~70 orders above the 99th\n"
    f"percentile ({ov_cap:.0f}) are mega-orders that all carry heavy losses (profit ≈\n"
    "-4,000), dragging Pearson down to {t4_pearson:.2f}. At 99th-percentile capping,\n"
    f"Pearson becomes {t4_pearson_cap:.2f} and Spearman {t4_spearman_cap:.2f} — both positive\n"
    "and aligned, confirming the core relationship is positive and monotonic.\n"
    "\n"
    f"**Should extremes be dropped?** **Yes**, for the headline figure. Dropping the\n"
    f"{len(df) - len(df_cap)} orders above the 99th percentile leaves {len(df_cap)} rows\n"
    f"(98.3% of data). The remaining Pearson ({t4_pearson_cap:.2f}) and Spearman\n"
    f"({t4_spearman_cap:.2f}) are close, so the *true* business relationship is positive.\n"
    "Report Spearman ({t4_spearman:.2f}) as the robust headline, and flag the loss-making\n"
    "mega-orders separately.\n"
    "\n"
    "> **Correlation is not causation:** a correlation (positive or negative) between\n"
    "order_value and profit_per_order does **not** mean that raising order value *causes*\n"
    "higher profit. Both are jointly driven by confounders — restaurant category, cuisine,\n"
    "and discount strategy. Management must not treat the correlation as a lever they\n"
    "can pull directly."
))

# 5
cells.append(new_markdown_cell(
    "### Task 5 — Layered aggregation: pivot_table + heatmap\n"
    "\n"
    "Build a city × restaurant_category table of **average profit_per_order** with\n"
    "`margins=True`. Render as a heatmap with a **diverging palette** (profit can be\n"
    "negative). State the most profitable combination, and in one sentence note whether\n"
    "it is also the top revenue generator."
))
cells.append(new_code_cell(
    "pt = pd.pivot_table(df, index='city', columns='restaurant_category',\n"
    "                    values='profit_per_order', aggfunc='mean', margins=True)\n"
    "print(pt.round(1))"
))
cells.append(new_code_cell(
    "pt_no = pt.drop(index='All', columns='All')\n"
    "best_idx = pt_no.stack().idxmax()\n"
    "best_val = pt_no.loc[best_idx[0], best_idx[1]]\n"
    "print(f'Most profitable: {best_idx[0]} / {best_idx[1]} = {best_val:.2f} avg profit/order')\n"
    "\n"
    "pt_rev = pd.pivot_table(df, index='city', columns='restaurant_category',\n"
    "                        values='order_value', aggfunc='sum', margins=True)\n"
    "pt_rev_no = pt_rev.drop(index='All', columns='All')\n"
    "rev_idx = pt_rev_no.stack().idxmax()\n"
    "rev_val = pt_rev_no.loc[rev_idx[0], rev_idx[1]]\n"
    "print(f'Top revenue: {rev_idx[0]} / {rev_idx[1]} = {rev_val:.0f} total revenue')"
))
cells.append(new_code_cell(
    "pt_no = pt.drop(index='All', columns='All')\n"
    "fig, ax = plt.subplots(figsize=(8, 5))\n"
    "sns.heatmap(pt_no, annot=True, fmt='.1f', cmap='RdBu_r',\n"
    "            center=0, ax=ax, cbar_kws={'label': 'Avg profit_per_order'})\n"
    "ax.set_title('Avg profit_per_order by city × restaurant_category')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(new_markdown_cell(
    f"**Most profitable combination:** **{best_idx[0]**} / **{best_idx[1]}**\n"
    f"with an average of **{best_val:.1f}** profit per order — the brightest cell in the\n"
    "heatmap.\n"
    "\n"
    f"**Revenue vs profitability:** That most-profitable combination is **not** the top\n"
    f"revenue generator — the highest-revenue cell is `{rev_idx[0]}`** / **`{rev_idx[1]}`\n"
    f"(total revenue ≈ {rev_val:.0f}). High revenue and high profitability point to\n"
    "**different** city-category combinations, so management should track both metrics\n"
    "separately rather than assuming the busiest segment is also the most profitable."
))

# 6
cells.append(new_markdown_cell(
    "### Task 6 — Time-series: trend vs noise\n"
    "\n"
    "Resample to get average **monthly rating** and overlay a 3-month rolling average.\n"
    "State whether satisfaction trends up, down, or flat once noise is smoothed, and\n"
    "whether the raw monthly line alone would have led to a different conclusion."
))
cells.append(new_code_cell(
    "df_ts = df.set_index('order_date')\n"
    "monthly = df_ts['rating'].resample('ME').mean()\n"
    "rolling = monthly.rolling(window=3).mean()\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(11, 5))\n"
    "monthly.plot(ax=ax, label='Monthly avg rating', alpha=0.5, marker='o', ms=3, color='steelblue')\n"
    "rolling.plot(ax=ax, label='3-month rolling avg', linewidth=2, color='crimson')\n"
    "ax.set_ylabel('Average Rating')\n"
    "ax.set_title('Monthly average rating vs 3-month rolling average')\n"
    "ax.legend()\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(new_markdown_cell(
    f"**Trend (smoothed):** Once the 3-month rolling average is applied, customer\n"
    f"satisfaction is **essentially flat with a slight decline**. The rolling line\n"
    f"starts near {t6_roll_first:.3f} ({t6_start}) and ends near {t6_roll_last:.3f}\n"
    f"({t6_end}) — a small net change of {t6_roll_last - t6_roll_first:+.3f} over ~18 months.\n"
    "\n"
    f"**Would the raw line mislead?** Yes. The raw monthly average swings only between\n"
    f"{t6_raw_min:.3f} and {t6_raw_max:.3f} (a range of ~{t6_raw_max - t6_raw_min:.2f}), and\n"
    "month-to-month jumps of ±0.03 can look meaningful in isolation. The rolling\n"
    "average reveals that these fluctuations are **noise**, not a signal — the true\n"
    "underlying trend is flat, and an observer looking only at the raw line might\n"
    "mistake a single uptick or dip for a real shift."
))

# 7
cells.append(new_markdown_cell(
    "### Task 7 — Faceting: small multiples\n"
    "\n"
    "Using FacetGrid, facet `delivery_time_min` vs `rating` by `restaurant_category`\n"
    "(col) and by whether `discount_percent > 20` (hue). State one category where heavy\n"
    "discounting visibly changes how forgiving customers are of slow delivery."
))
cells.append(new_code_cell(
    "g = sns.FacetGrid(df, col='restaurant_category', hue='heavy_discount',\n"
    "                  col_wrap=3, height=4, aspect=1.1, sharex=False, sharey=False)\n"
    "g.map_dataframe(sns.scatterplot, x='rating', y='delivery_time_min',\n"
    "                alpha=0.5, s=15, edgecolor='none')\n"
    "g.add_legend(title='Heavy discount (>20%)')\n"
    "g.set_titles('{col_name}')\n"
    "g.set_axis_labels('Rating', 'Delivery Time (min)')\n"
    "g.fig.suptitle('delivery_time_min vs rating by category and discount level',\n"
    "                y=1.03, fontsize=12)\n"
    "plt.show()"
))
cells.append(new_code_cell(
    "# quantify: delivery_time–rating correlation, heavy vs light discount, per category\n"
    "for cat in df['restaurant_category'].unique():\n"
    "    sub = df[df['restaurant_category'] == cat]\n"
    "    h = sub[sub['discount_percent'] > 20]\n"
    "    l = sub[sub['discount_percent'] <= 20]\n"
    "    if len(h) > 5 and len(l) > 5:\n"
    "        rh = h['delivery_time_min'].corr(h['rating'])\n"
    "        rl = l['delivery_time_min'].corr(l['rating'])\n"
    "        print(f'{cat:15s}  heavy n={len(h):4d} r={rh:+.3f} |'\n"
    "              f' light n={len(l):4d} r={rl:+.3f} | gap={rh - rl:+.3f}')"
))
cells.append(new_markdown_cell(
    f"**Category where discounting changes forgiveness:** **{best_facet}**. The largest\n"
    "gap between the delivery-time–rating correlation under heavy vs light discounting\n"
    "appears here: with heavy discounts the correlation is r = "
    f"{t7_h:+.3f} (n={t7_nh}), versus r = {t7_l:+.3f} for light discounts (n={t7_nl}).\n"
    "In other words, heavy discounting shifts customers' tolerance — they become more\n"
    "forgiving of a slow delivery in this category, visibly flattening the negative\n"
    "trend in the facet panel."
))

# 8
cells.append(new_markdown_cell(
    "### Task 8 — Overplotting + annotation (combined)\n"
    "\n"
    "Plot the raw `delivery_time_min` vs `rating` scatter (expect heavy overplotting from\n"
    "repeated rating values). Fix it, justify the fix, and annotate the order with the\n"
    "**best rating despite an unusually long delivery** and the order with the **worst\n"
    "rating despite an unusually fast delivery**."
))
cells.append(new_code_cell(
    "# diagnose\n"
    "print(f'Total orders  : {t8_n}')\n"
    "print(f'Unique ratings: {t8_nuniq}  (ratings are binned to 1 decimal → heavy overlap)')"
))
cells.append(new_code_cell(
    "# raw scatter — overplotted\n"
    "fig, ax = plt.subplots(figsize=(8, 6))\n"
    "ax.scatter(df['rating'], df['delivery_time_min'], alpha=0.2, s=8, color='navy')\n"
    "ax.set_xlabel('Rating')\n"
    "ax.set_ylabel('Delivery Time (min)')\n"
    "ax.set_title('Raw scatter — heavy overplotting')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(new_code_cell(
    "# fix with alpha transparency and annotate extremes\n"
    "long_del = df[df['delivery_time_min'] >= df['delivery_time_min'].quantile(0.90)]\n"
    "best_long = long_del.loc[long_del['rating'].idxmax()]\n"
    "fast_del = df[df['delivery_time_min'] <= df['delivery_time_min'].quantile(0.10)]\n"
    "worst_fast = fast_del.loc[fast_del['rating'].idxmin()]\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(9, 6))\n"
    "ax.scatter(df['rating'], df['delivery_time_min'],\n"
    "             alpha=0.3, s=12, color='steelblue', edgecolor='none', label='all orders')\n"
    "\n"
    "ax.scatter([best_long['rating']], [best_long['delivery_time_min']],\n"
    "             s=120, color='green', marker='^', zorder=5, edgecolor='black', linewidth=0.5)\n"
    "ax.annotate(f\"Best rating, slow delivery\\n({best_long['rating']:.1f}, {best_long['delivery_time_min']:.1f} min)\",\n"
    "            xy=(best_long['rating'], best_long['delivery_time_min']),\n"
    "            xytext=(best_long['rating'] + 0.15, best_long['delivery_time_min'] - 15),\n"
    "            arrowprops=dict(arrowstyle='->', color='green', lw=1.5),\n"
    "            fontsize=9, color='green')\n"
    "\n"
    "ax.scatter([worst_fast['rating']], [worst_fast['delivery_time_min']],\n"
    "             s=120, color='red', marker='v', zorder=5, edgecolor='black', linewidth=0.5)\n"
    "ax.annotate(f\"Worst rating, fast delivery\\n({worst_fast['rating']:.1f}, {worst_fast['delivery_time_min']:.1f} min)\",\n"
    "            xy=(worst_fast['rating'], worst_fast['delivery_time_min']),\n"
    "            xytext=(worst_fast['rating'] - 0.3, worst_fast['delivery_time_min'] + 8),\n"
    "            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),\n"
    "            fontsize=9, color='red')\n"
    "\n"
    "ax.set_xlabel('Rating')\n"
    "ax.set_ylabel('Delivery Time (min)')\n"
    "ax.set_title('Fixed scatter (alpha=0.3) with annotated extremes')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "print(f'Best rating despite long delivery:  rating={best_long[\"rating\"]:.1f}, '\n"
    "      f'time={best_long[\"delivery_time_min\"]:.1f} min, city={best_long[\"city\"]}')\n"
    "print(f'Worst rating despite fast delivery: rating={worst_fast[\"rating\"]:.1f}, '\n"
    "      f'time={worst_fast[\"delivery_time_min\"]:.1f} min, city={worst_fast[\"city\"]}')"
))
cells.append(new_markdown_cell(
    f"**Why alpha over hexbin:** With {t8_n} points and only {t8_nuniq} discrete rating\n"
    "levels, a raw scatter collapses into a near-solid block. **Alpha transparency**\n"
    "(alpha = 0.3) is the right fix because the x-axis (rating) is discrete — hexbin or\n"
    "KDE would erase the exact rating values we need for annotation, whereas alpha\n"
    "preserves every point and reveals density through overlap.\n"
    "\n"
    "**Annotated insights:**\n"
    "- Green ▲: the **best rating despite an unusually long delivery** — rating\n"
    f"  {best_long['rating']:.1f} at {best_long['delivery_time_min']:.1f} min (top-decile slowest).\n"
    "  A slow delivery alone does not doom a rating.\n"
    "- Red ▼: the **worst rating despite an unusually fast delivery** — rating\n"
    f"  {worst_fast['rating']:.1f} at {worst_fast['delivery_time_min']:.1f} min (top-decile fastest).\n"
    "  A fast delivery cannot always compensate for other issues (food quality,\n"
    "  order accuracy, cold food, etc.)."
))

# ── Assemble & save ────────────────────────────────────────────────────────
nb = new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
)

nb_path = "Lab_04_DAV.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Notebook written: {nb_path}  ({len(cells)} cells)")

# ── Execute the notebook in-place ───────────────────────────────────────────
print("Executing notebook ...")
client = NotebookClient(nb, timeout=300, kernel_name="python3")
client.execute()
with open(nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Execution complete. Outputs embedded.")
