(function () {
  "use strict";

  const TABULATOR_CSS_ID = "tn-results-browser-tabulator-css";
  const TABULATOR_CSS_URL =
    "https://cdn.jsdelivr.net/npm/tabulator-tables@6.3.1/dist/css/tabulator.min.css";

  const COLORS = {
    accent: "#0f766e",
    accentStrong: "#115e59",
    accentSoft: "#dff4ef",
    ink: "#10212b",
    line: "#c7d8d4",
    gold: "#b45309",
    goldSoft: "#fef3c7",
    blue: "#2563eb",
    rose: "#be123c",
    slate: "#475569",
    sand: "#f7f4ec",
  };

  const state = {
    root: null,
    activeTab: "benchmark",
    benchmark: null,
    benchmarkFilters: {
      family: "all",
      campaign: "all",
      tier: "all",
      domain: "all",
      platform: "all",
      datasetQuery: "",
      profileQuery: "",
      selectedDataset: "",
      tableMode: "datasets",
    },
    tabarena: null,
    tabarenaLoaded: false,
    tabarenaFilters: {
      problemType: "all",
      query: "",
      tableMode: "per_dataset",
    },
    charts: {},
    tables: {},
    resizeBound: false,
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function readText(value) {
    if (value === null || value === undefined) {
      return "";
    }
    return String(value);
  }

  function escapeHtml(value) {
    const text = readText(value);
    const el = document.createElement("span");
    el.textContent = text;
    return el.innerHTML;
  }

  function normalize(value) {
    return readText(value).trim().toLowerCase();
  }

  function formatInt(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return "—";
    }
    return number.toLocaleString();
  }

  function formatFloat(value, digits) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return "—";
    }
    return number.toFixed(digits);
  }

  function formatPercent(value, digits) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return "—";
    }
    return `${(number * 100).toFixed(digits)}%`;
  }

  function mean(values) {
    if (!values.length) {
      return null;
    }
    const total = values.reduce((sum, value) => sum + value, 0);
    return total / values.length;
  }

  function median(values) {
    if (!values.length) {
      return null;
    }
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    if (sorted.length % 2) {
      return sorted[mid];
    }
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }

  function setStatus(message, tone) {
    const node = byId("tn-browser-status");
    if (!node) {
      return;
    }
    node.textContent = message;
    node.classList.remove("is-error", "is-ready");
    if (tone === "error") {
      node.classList.add("is-error");
    } else if (tone === "ready") {
      node.classList.add("is-ready");
    }
  }

  function ensureTabulatorStyles() {
    if (document.getElementById(TABULATOR_CSS_ID)) {
      return;
    }
    const link = document.createElement("link");
    link.id = TABULATOR_CSS_ID;
    link.rel = "stylesheet";
    link.href = TABULATOR_CSS_URL;
    document.head.appendChild(link);
  }

  function unpackTable(table) {
    if (!table || !Array.isArray(table.columns) || !Array.isArray(table.rows)) {
      return [];
    }
    return table.rows.map((row) => {
      const record = {};
      table.columns.forEach((column, index) => {
        record[column] = row[index];
      });
      return record;
    });
  }

  async function fetchJson(url) {
    const response = await fetch(url, { credentials: "same-origin" });
    if (!response.ok) {
      throw new Error(`Failed to fetch ${url}: ${response.status}`);
    }
    return response.json();
  }

  function matchesQuery(parts, query) {
    if (!query) {
      return true;
    }
    return parts.some((part) => normalize(part).includes(query));
  }

  function tierLabel(value) {
    const map = {
      easy: "Easy",
      medium: "Medium",
      hard: "Hard",
      very_hard: "Very Hard",
    };
    return map[readText(value)] || readText(value) || "Unknown";
  }

  function sotaColor(status) {
    if (status === "above") {
      return COLORS.accent;
    }
    if (status === "within") {
      return COLORS.blue;
    }
    return COLORS.rose;
  }

  function metricCard(label, value, sublabel) {
    return `
      <article class="tn-metric-card">
        <div class="tn-metric-label">${escapeHtml(label)}</div>
        <div class="tn-metric-value">${escapeHtml(value)}</div>
        <div class="tn-metric-sub">${escapeHtml(sublabel || "")}</div>
      </article>
    `;
  }

  function ensureChart(key, elementId) {
    if (state.charts[key]) {
      return state.charts[key];
    }
    const element = byId(elementId);
    if (!element) {
      return null;
    }
    const chart = echarts.init(element, null, { renderer: "canvas" });
    state.charts[key] = chart;
    if (!state.resizeBound) {
      window.addEventListener("resize", () => {
        Object.values(state.charts).forEach((instance) => instance.resize());
      });
      state.resizeBound = true;
    }
    return chart;
  }

  function renderEmptyChart(chart, message) {
    chart.clear();
    chart.setOption({
      animation: false,
      title: {
        text: message,
        left: "center",
        top: "middle",
        textStyle: {
          color: COLORS.slate,
          fontSize: 14,
          fontWeight: 500,
        },
      },
    });
  }

  function inflateBenchmarkPayload(payload) {
    return {
      metadata: payload.metadata || { available: false },
      datasets: unpackTable(payload.datasets),
      profiles: unpackTable(payload.profiles),
      datasetProfiles: unpackTable(payload.dataset_profiles),
      runs: unpackTable(payload.runs),
    };
  }

  function inflateTabarenaPayload(payload) {
    return {
      metadata: payload.metadata || { available: false },
      summary: payload.summary || {},
      leaderboardOverall: unpackTable(payload.leaderboard_overall),
      leaderboardBinary: unpackTable(payload.leaderboard_binary),
      leaderboardMulticlass: unpackTable(payload.leaderboard_multiclass),
      perDataset: unpackTable(payload.per_dataset),
    };
  }

  function populateSelect(selectId, values, formatter) {
    const select = byId(selectId);
    if (!select) {
      return;
    }
    const current = select.value;
    const options = ['<option value="all">All</option>'];
    values.forEach((value) => {
      const label = formatter ? formatter(value) : value;
      options.push(`<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`);
    });
    select.innerHTML = options.join("");
    if ([...select.options].some((option) => option.value === current)) {
      select.value = current;
    }
  }

  function initializeBenchmarkFilters() {
    const filters = (state.benchmark && state.benchmark.metadata && state.benchmark.metadata.filters) || {};
    const familyLabels = new Map(
      (state.benchmark.datasetProfiles || []).map((row) => [row.family, row.family_label || row.family])
    );
    const campaignLabels = new Map(
      (state.benchmark.datasetProfiles || []).map((row) => [row.campaign, row.campaign_label || row.campaign])
    );
    populateSelect("tn-benchmark-family", filters.families || [], (value) => `${value} · ${familyLabels.get(value) || value}`);
    populateSelect(
      "tn-benchmark-campaign",
      filters.campaigns || [],
      (value) => campaignLabels.get(value) || value
    );
    populateSelect("tn-benchmark-tier", filters.tiers || [], tierLabel);
    populateSelect("tn-benchmark-domain", filters.domains || [], (value) => value);
    populateSelect("tn-benchmark-platform", filters.platforms || [], (value) => value);
  }

  function buildFilteredDatasets(datasetProfiles) {
    const groups = new Map();
    datasetProfiles.forEach((row) => {
      const existing = groups.get(row.dataset_id);
      const bestCandidate = !existing || Number(row.mean_ba) > Number(existing.best_ba);
      if (!existing) {
        groups.set(row.dataset_id, {
          dataset_id: row.dataset_id,
          dataset_label: row.dataset_label,
          tier: row.tier,
          domain: row.domain,
          platform: row.platform,
          n_samples_total: row.n_samples_total,
          n_features_total: row.n_features_total,
          best_ba: row.mean_ba,
          top_profile: row.profile,
          top_family: row.family,
          top_family_label: row.family_label,
          top_campaign: row.campaign,
          top_campaign_label: row.campaign_label,
          sota_low: null,
          sota_high: null,
          sota_confidence: "",
          sota_status: "within",
          delta_vs_sota_high: null,
          profile_count: 1,
          baValues: [Number(row.mean_ba)],
          profiles: new Set([row.profile]),
        });
        return;
      }

      existing.baValues.push(Number(row.mean_ba));
      existing.profiles.add(row.profile);
      existing.profile_count = existing.profiles.size;
      if (bestCandidate) {
        existing.best_ba = row.mean_ba;
        existing.top_profile = row.profile;
        existing.top_family = row.family;
        existing.top_family_label = row.family_label;
        existing.top_campaign = row.campaign;
        existing.top_campaign_label = row.campaign_label;
      }
    });

    const metaById = new Map((state.benchmark.datasets || []).map((row) => [row.dataset_id, row]));
    return [...groups.values()]
      .map((row) => {
        const meta = metaById.get(row.dataset_id) || {};
        const sotaLow = Number(meta.sota_low);
        const sotaHigh = Number(meta.sota_high);
        const bestBa = Number(row.best_ba);
        let sotaStatus = "within";
        if (Number.isFinite(sotaHigh) && bestBa > sotaHigh) {
          sotaStatus = "above";
        } else if (Number.isFinite(sotaLow) && bestBa < sotaLow) {
          sotaStatus = "below";
        }
        return {
          dataset_id: row.dataset_id,
          dataset_label: row.dataset_label,
          tier: row.tier,
          domain: row.domain,
          platform: row.platform,
          n_samples_total: row.n_samples_total,
          n_features_total: row.n_features_total,
          best_ba: bestBa,
          median_profile_ba: median(row.baValues),
          top_profile: row.top_profile,
          top_family: row.top_family,
          top_family_label: row.top_family_label,
          top_campaign: row.top_campaign,
          top_campaign_label: row.top_campaign_label,
          profile_count: row.profile_count,
          sota_low: Number.isFinite(sotaLow) ? sotaLow : null,
          sota_high: Number.isFinite(sotaHigh) ? sotaHigh : null,
          sota_confidence: meta.sota_confidence || "",
          sota_status: sotaStatus,
          delta_vs_sota_high: Number.isFinite(sotaHigh) ? bestBa - sotaHigh : null,
        };
      })
      .sort((a, b) => (Number(b.best_ba) - Number(a.best_ba)) || a.dataset_label.localeCompare(b.dataset_label));
  }

  function buildFilteredProfiles(datasetProfiles) {
    const groups = new Map();
    datasetProfiles.forEach((row) => {
      const key = row.profile;
      const existing = groups.get(key);
      if (!existing) {
        groups.set(key, {
          profile: row.profile,
          family: row.family,
          family_label: row.family_label,
          campaign: row.campaign,
          campaign_label: row.campaign_label,
          values: [Number(row.mean_ba)],
          datasets: new Set([row.dataset_id]),
        });
        return;
      }
      existing.values.push(Number(row.mean_ba));
      existing.datasets.add(row.dataset_id);
    });

    return [...groups.values()]
      .map((row) => ({
        profile: row.profile,
        family: row.family,
        family_label: row.family_label,
        campaign: row.campaign,
        campaign_label: row.campaign_label,
        mean_ba: mean(row.values),
        median_ba: median(row.values),
        best_ba: Math.max(...row.values),
        dataset_count: row.datasets.size,
      }))
      .sort((a, b) => (Number(b.mean_ba) - Number(a.mean_ba)) || a.profile.localeCompare(b.profile));
  }

  function filterBenchmarkData() {
    if (!state.benchmark) {
      return { datasets: [], profiles: [], datasetProfiles: [], runs: [] };
    }

    const f = state.benchmarkFilters;
    const datasetQuery = normalize(f.datasetQuery);
    const profileQuery = normalize(f.profileQuery);

    const datasetProfiles = state.benchmark.datasetProfiles.filter((row) => {
      if (f.family !== "all" && row.family !== f.family) {
        return false;
      }
      if (f.campaign !== "all" && row.campaign !== f.campaign) {
        return false;
      }
      if (f.tier !== "all" && row.tier !== f.tier) {
        return false;
      }
      if (f.domain !== "all" && row.domain !== f.domain) {
        return false;
      }
      if (f.platform !== "all" && row.platform !== f.platform) {
        return false;
      }
      if (!matchesQuery([row.dataset_label, row.dataset_id], datasetQuery)) {
        return false;
      }
      if (!matchesQuery([row.profile, row.family_label, row.campaign_label], profileQuery)) {
        return false;
      }
      return true;
    });

    const datasets = buildFilteredDatasets(datasetProfiles);
    const profiles = buildFilteredProfiles(datasetProfiles);
    const allowedDatasets = new Set(datasets.map((row) => row.dataset_id));
    const allowedProfiles = new Set(profiles.map((row) => row.profile));
    const runs = state.benchmark.runs.filter(
      (row) => allowedDatasets.has(row.dataset_id) && allowedProfiles.has(row.profile)
    );

    if (!datasets.some((row) => row.dataset_id === state.benchmarkFilters.selectedDataset)) {
      state.benchmarkFilters.selectedDataset = datasets.length ? datasets[0].dataset_id : "";
    }

    return { datasets, profiles, datasetProfiles, runs };
  }

  function renderBenchmarkMetrics(filtered) {
    const container = byId("tn-benchmark-metrics");
    if (!container) {
      return;
    }
    const sotaCounts = filtered.datasets.reduce(
      (acc, row) => {
        acc[row.sota_status] = (acc[row.sota_status] || 0) + 1;
        return acc;
      },
      { above: 0, within: 0, below: 0 }
    );
    const meanBa = mean(filtered.datasetProfiles.map((row) => Number(row.mean_ba)));
    const bestDataset = filtered.datasets[0];

    container.innerHTML = [
      metricCard("Datasets", formatInt(filtered.datasets.length), "Datasets remaining after filters"),
      metricCard("Profiles", formatInt(filtered.profiles.length), "Unique profiles in the filtered slice"),
      metricCard("Profile-Dataset Cells", formatInt(filtered.datasetProfiles.length), "Aggregated dataset/profile comparisons"),
      metricCard("Mean BA", formatFloat(meanBa, 3), "Average filtered dataset/profile balanced accuracy"),
      metricCard(
        "Best Dataset",
        bestDataset ? `${bestDataset.dataset_label}` : "—",
        bestDataset ? `Best BA ${formatFloat(bestDataset.best_ba, 3)}` : "No filtered result rows"
      ),
      metricCard(
        "SOTA Split",
        `${sotaCounts.above} / ${sotaCounts.within} / ${sotaCounts.below}`,
        "Above / within / below for the filtered best-per-dataset slice"
      ),
    ].join("");
  }

  function renderBenchmarkSelection(filtered) {
    const node = byId("tn-benchmark-selection");
    if (!node) {
      return;
    }
    const selected = filtered.datasets.find((row) => row.dataset_id === state.benchmarkFilters.selectedDataset);
    if (!selected) {
      node.textContent = "No dataset selected.";
      return;
    }
    node.textContent = `Focused dataset: ${selected.dataset_label} · top profile ${selected.top_profile} · best BA ${formatFloat(
      selected.best_ba,
      3
    )}`;
  }

  function renderBenchmarkDatasetChart(filtered) {
    const chart = ensureChart("benchmarkDataset", "tn-benchmark-dataset-chart");
    if (!chart) {
      return;
    }
    if (!filtered.datasets.length) {
      renderEmptyChart(chart, "No datasets match the current filters.");
      return;
    }

    chart.off("click");
    chart.on("click", (event) => {
      if (!event.data || !event.data.datasetId) {
        return;
      }
      state.benchmarkFilters.selectedDataset = event.data.datasetId;
      renderBenchmark();
    });

    const data = filtered.datasets.map((row) => ({
      name: row.dataset_label,
      value: [Number(row.n_samples_total), Number(row.best_ba)],
      datasetId: row.dataset_id,
      symbolSize: Math.max(12, Math.min(42, Math.log10(Number(row.n_features_total) + 10) * 7)),
      itemStyle: { color: sotaColor(row.sota_status) },
      row,
    }));

    chart.setOption({
      backgroundColor: "transparent",
      animationDuration: 300,
      grid: { left: 56, right: 24, top: 24, bottom: 52 },
      tooltip: {
        trigger: "item",
        formatter(params) {
          const row = params.data.row;
          return `
            <strong>${escapeHtml(row.dataset_label)}</strong><br/>
            Tier: ${escapeHtml(tierLabel(row.tier))}<br/>
            Samples / features: ${formatInt(row.n_samples_total)} / ${formatInt(row.n_features_total)}<br/>
            Best BA: ${formatFloat(row.best_ba, 3)}<br/>
            Top profile: ${escapeHtml(row.top_profile)}<br/>
            SOTA status: ${escapeHtml(row.sota_status)}
          `;
        },
      },
      xAxis: {
        type: "log",
        name: "Samples",
        nameLocation: "middle",
        nameGap: 34,
        axisLine: { lineStyle: { color: COLORS.line } },
        splitLine: { lineStyle: { color: "rgba(16,33,43,0.08)" } },
      },
      yAxis: {
        type: "value",
        min: 0.25,
        max: 1.02,
        name: "Best BA",
        nameLocation: "middle",
        nameGap: 44,
        axisLine: { lineStyle: { color: COLORS.line } },
        splitLine: { lineStyle: { color: "rgba(16,33,43,0.08)" } },
      },
      series: [
        {
          type: "scatter",
          data,
          emphasis: {
            focus: "series",
            itemStyle: {
              borderColor: COLORS.ink,
              borderWidth: 1.5,
            },
          },
        },
      ],
    });
  }

  function renderBenchmarkFamilyChart(filtered) {
    const chart = ensureChart("benchmarkFamily", "tn-benchmark-family-chart");
    if (!chart) {
      return;
    }
    if (!filtered.profiles.length) {
      renderEmptyChart(chart, "No families remain after filtering.");
      return;
    }

    const familyGroups = new Map();
    filtered.profiles.forEach((row) => {
      const key = row.family_label;
      const existing = familyGroups.get(key) || { family_label: key, values: [], datasetCount: 0 };
      existing.values.push(Number(row.mean_ba));
      existing.datasetCount += Number(row.dataset_count);
      familyGroups.set(key, existing);
    });

    const rows = [...familyGroups.values()]
      .map((row) => ({
        family_label: row.family_label,
        mean_ba: mean(row.values),
        dataset_count: row.datasetCount,
      }))
      .sort((a, b) => Number(a.mean_ba) - Number(b.mean_ba));

    chart.setOption({
      backgroundColor: "transparent",
      animationDuration: 300,
      grid: { left: 108, right: 28, top: 24, bottom: 24 },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        formatter(params) {
          const row = params[0].data.row;
          return `<strong>${escapeHtml(row.family_label)}</strong><br/>Mean BA: ${formatFloat(row.mean_ba, 3)}<br/>Profile-dataset coverage: ${formatInt(
            row.dataset_count
          )}`;
        },
      },
      xAxis: {
        type: "value",
        min: 0.55,
        max: 1.0,
        axisLine: { lineStyle: { color: COLORS.line } },
        splitLine: { lineStyle: { color: "rgba(16,33,43,0.08)" } },
      },
      yAxis: {
        type: "category",
        data: rows.map((row) => row.family_label),
        axisTick: { show: false },
        axisLine: { lineStyle: { color: COLORS.line } },
      },
      series: [
        {
          type: "bar",
          data: rows.map((row) => ({
            value: Number(row.mean_ba),
            row,
            itemStyle: { color: COLORS.gold },
          })),
          barWidth: 20,
          showBackground: true,
          backgroundStyle: { color: "rgba(180,83,9,0.08)" },
        },
      ],
    });
  }

  function renderBenchmarkDetailChart(filtered) {
    const chart = ensureChart("benchmarkDetail", "tn-benchmark-detail-chart");
    if (!chart) {
      return;
    }

    let rows = [];
    const selectedDataset = state.benchmarkFilters.selectedDataset;
    if (selectedDataset) {
      rows = filtered.datasetProfiles
        .filter((row) => row.dataset_id === selectedDataset)
        .sort((a, b) => Number(b.mean_ba) - Number(a.mean_ba))
        .slice(0, 20);
    }
    if (!rows.length) {
      rows = filtered.profiles.slice(0, 20).map((row) => ({
        profile: row.profile,
        mean_ba: row.mean_ba,
        family_label: row.family_label,
        dataset_label: "Filtered frontier",
      }));
    }
    if (!rows.length) {
      renderEmptyChart(chart, "No profile detail rows remain after filtering.");
      return;
    }

    chart.setOption({
      backgroundColor: "transparent",
      animationDuration: 300,
      grid: { left: 136, right: 28, top: 24, bottom: 42 },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        formatter(params) {
          const row = params[0].data.row;
          return `<strong>${escapeHtml(row.profile)}</strong><br/>Mean BA: ${formatFloat(row.mean_ba, 3)}<br/>${escapeHtml(
            row.family_label
          )}`;
        },
      },
      xAxis: {
        type: "value",
        min: 0.4,
        max: 1.0,
        axisLine: { lineStyle: { color: COLORS.line } },
        splitLine: { lineStyle: { color: "rgba(16,33,43,0.08)" } },
      },
      yAxis: {
        type: "category",
        data: rows.map((row) => row.profile),
        axisTick: { show: false },
        axisLine: { lineStyle: { color: COLORS.line } },
      },
      dataZoom: rows.length > 12 ? [{ type: "inside", yAxisIndex: 0 }] : [],
      series: [
        {
          type: "bar",
          data: rows
            .slice()
            .reverse()
            .map((row) => ({
              value: Number(row.mean_ba),
              row,
              itemStyle: { color: COLORS.accent },
            })),
          barWidth: 18,
        },
      ],
    });
  }

  function percentCellFormatter(digits) {
    return function (cell) {
      const value = cell.getValue();
      if (value === null || value === undefined || value === "") {
        return "—";
      }
      return formatFloat(value, digits);
    };
  }

  function textCellFormatter(cell) {
    return readText(cell.getValue()) || "—";
  }

  function intCellFormatter(cell) {
    return formatInt(cell.getValue());
  }

  function benchmarkTableColumns(mode) {
    if (mode === "profiles") {
      return [
        { title: "Profile", field: "profile", minWidth: 180, formatter: textCellFormatter },
        { title: "Family", field: "family_label", minWidth: 140, formatter: textCellFormatter },
        { title: "Campaign", field: "campaign_label", minWidth: 160, formatter: textCellFormatter },
        { title: "Mean BA", field: "mean_ba", hozAlign: "right", formatter: percentCellFormatter(3) },
        { title: "Median BA", field: "median_ba", hozAlign: "right", formatter: percentCellFormatter(3) },
        { title: "Best BA", field: "best_ba", hozAlign: "right", formatter: percentCellFormatter(3) },
        { title: "Datasets", field: "dataset_count", hozAlign: "right", formatter: textCellFormatter },
      ];
    }
    if (mode === "runs") {
      return [
        { title: "Dataset", field: "dataset_label", minWidth: 220, formatter: textCellFormatter },
        { title: "Profile", field: "profile", minWidth: 170, formatter: textCellFormatter },
        { title: "Family", field: "family_label", minWidth: 130, formatter: textCellFormatter },
        { title: "Campaign", field: "campaign_label", minWidth: 150, formatter: textCellFormatter },
        { title: "Seed", field: "seed", hozAlign: "right", formatter: intCellFormatter },
        { title: "BA", field: "balanced_accuracy", hozAlign: "right", formatter: percentCellFormatter(3) },
        { title: "Macro-F1", field: "macro_f1", hozAlign: "right", formatter: percentCellFormatter(3) },
        { title: "ROC AUC", field: "roc_auc", hozAlign: "right", formatter: percentCellFormatter(3) },
        { title: "Model", field: "model", minWidth: 180, formatter: textCellFormatter },
      ];
    }
    return [
      { title: "Dataset", field: "dataset_label", minWidth: 220, formatter: textCellFormatter },
      { title: "Tier", field: "tier", minWidth: 110, formatter: (cell) => tierLabel(cell.getValue()) },
      { title: "Domain", field: "domain", minWidth: 110, formatter: textCellFormatter },
      { title: "Samples", field: "n_samples_total", hozAlign: "right", formatter: intCellFormatter },
      { title: "Features", field: "n_features_total", hozAlign: "right", formatter: intCellFormatter },
      { title: "Best BA", field: "best_ba", hozAlign: "right", formatter: percentCellFormatter(3) },
      { title: "Top Profile", field: "top_profile", minWidth: 170, formatter: textCellFormatter },
      { title: "SOTA", field: "sota_status", minWidth: 110, formatter: textCellFormatter },
      { title: "Delta vs High", field: "delta_vs_sota_high", hozAlign: "right", formatter: percentCellFormatter(3) },
    ];
  }

  function ensureBenchmarkTable() {
    if (state.tables.benchmark) {
      return state.tables.benchmark;
    }
    const table = new Tabulator(byId("tn-benchmark-table"), {
      layout: "fitColumns",
      pagination: true,
      paginationSize: 12,
      movableColumns: true,
      placeholder: "No rows match the current filters.",
      initialSort: [{ column: "best_ba", dir: "desc" }],
      rowClick(_event, row) {
        const data = row.getData();
        if (data.dataset_id) {
          state.benchmarkFilters.selectedDataset = data.dataset_id;
          renderBenchmark();
        }
      },
    });
    state.tables.benchmark = table;
    return table;
  }

  function renderBenchmarkTable(filtered) {
    const mode = state.benchmarkFilters.tableMode;
    const table = ensureBenchmarkTable();
    const data =
      mode === "profiles" ? filtered.profiles : mode === "runs" ? filtered.runs : filtered.datasets;
    table.setColumns(benchmarkTableColumns(mode));
    table.setPageSize(mode === "runs" ? 20 : 12);
    table.replaceData(data);
  }

  function renderBenchmark() {
    const filtered = filterBenchmarkData();
    renderBenchmarkMetrics(filtered);
    renderBenchmarkSelection(filtered);
    renderBenchmarkDatasetChart(filtered);
    renderBenchmarkFamilyChart(filtered);
    renderBenchmarkDetailChart(filtered);
    renderBenchmarkTable(filtered);
  }

  function filterTabarenaData() {
    if (!state.tabarena) {
      return {
        perDataset: [],
        leaderboardOverall: [],
        leaderboardBinary: [],
        leaderboardMulticlass: [],
      };
    }

    const query = normalize(state.tabarenaFilters.query);
    const problemType = state.tabarenaFilters.problemType;

    const perDataset = state.tabarena.perDataset.filter((row) => {
      if (problemType !== "all" && row.problem_type !== problemType) {
        return false;
      }
      if (!matchesQuery([row.dataset, row.ref_best_method, row.model], query)) {
        return false;
      }
      return true;
    });

    const filterLeaderboard = (rows) =>
      rows.filter((row) => matchesQuery([row.method], query)).sort((a, b) => Number(b.elo) - Number(a.elo));

    return {
      perDataset,
      leaderboardOverall: filterLeaderboard(state.tabarena.leaderboardOverall),
      leaderboardBinary: filterLeaderboard(state.tabarena.leaderboardBinary),
      leaderboardMulticlass: filterLeaderboard(state.tabarena.leaderboardMulticlass),
    };
  }

  function renderTabarenaMetrics(filtered) {
    const node = byId("tn-tabarena-metrics");
    if (!node || !state.tabarena) {
      return;
    }
    const meta = state.tabarena.metadata || {};
    const summary = state.tabarena.summary || {};
    node.innerHTML = [
      metricCard("Datasets", formatInt(filtered.perDataset.length), "Filtered TabArena dataset rows"),
      metricCard("Overall Elo", formatFloat(meta.overall_elo, 1), `Rank ${formatInt(summary.overall_rank_position)}`),
      metricCard(
        "Normalized Score",
        formatFloat(meta.overall_normalized_score, 3),
        `${formatFloat(summary.wins_vs_dataset_best || 0, 0)} wins vs dataset best`
      ),
      metricCard(
        "Binary / Multiclass Elo",
        `${formatFloat(meta.binary_elo, 1)} / ${formatFloat(meta.multiclass_elo, 1)}`,
        "Current leaderboard-style split"
      ),
    ].join("");
  }

  function renderTabarenaLeaderboardChart(filtered) {
    const chart = ensureChart("tabarenaLeaderboard", "tn-tabarena-leaderboard-chart");
    if (!chart) {
      return;
    }
    const rows = filtered.leaderboardOverall;
    if (!rows.length) {
      renderEmptyChart(chart, "No leaderboard rows match the current search.");
      return;
    }

    const reversed = rows.slice().reverse();
    chart.setOption({
      backgroundColor: "transparent",
      animationDuration: 300,
      grid: { left: 166, right: 28, top: 24, bottom: 30 },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        formatter(params) {
          const row = params[0].data.row;
          return `<strong>${escapeHtml(row.method)}</strong><br/>Elo: ${formatFloat(row.elo, 1)}<br/>Rank: ${formatFloat(
            row.rank,
            2
          )}<br/>Winrate: ${formatFloat(row.winrate, 3)}`;
        },
      },
      xAxis: {
        type: "value",
        axisLine: { lineStyle: { color: COLORS.line } },
        splitLine: { lineStyle: { color: "rgba(16,33,43,0.08)" } },
      },
      yAxis: {
        type: "category",
        data: reversed.map((row) => row.method),
        axisTick: { show: false },
        axisLine: { lineStyle: { color: COLORS.line } },
      },
      dataZoom: rows.length > 14 ? [{ type: "inside", yAxisIndex: 0 }] : [],
      series: [
        {
          type: "bar",
          barWidth: 18,
          data: reversed.map((row) => ({
            value: Number(row.elo),
            row,
            itemStyle: {
              color: normalize(row.method).includes("tabnetics") ? COLORS.accent : COLORS.gold,
            },
          })),
        },
      ],
    });
  }

  function renderTabarenaGapChart(filtered) {
    const chart = ensureChart("tabarenaGap", "tn-tabarena-gap-chart");
    if (!chart) {
      return;
    }
    const rows = filtered.perDataset.slice(0, 38).reverse();
    if (!rows.length) {
      renderEmptyChart(chart, "No TabArena datasets match the current filters.");
      return;
    }

    chart.setOption({
      backgroundColor: "transparent",
      animationDuration: 300,
      grid: { left: 176, right: 28, top: 24, bottom: 30 },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        formatter(params) {
          const row = params[0].data.row;
          return `<strong>${escapeHtml(row.dataset)}</strong><br/>Delta vs best: ${formatFloat(row.delta_vs_best, 4)}<br/>Best official: ${escapeHtml(
            row.ref_best_method
          )}<br/>Selected model: ${escapeHtml(row.model)}`;
        },
      },
      xAxis: {
        type: "value",
        axisLine: { lineStyle: { color: COLORS.line } },
        splitLine: { lineStyle: { color: "rgba(16,33,43,0.08)" } },
      },
      yAxis: {
        type: "category",
        data: rows.map((row) => row.dataset),
        axisTick: { show: false },
        axisLine: { lineStyle: { color: COLORS.line } },
      },
      dataZoom: rows.length > 14 ? [{ type: "inside", yAxisIndex: 0 }] : [],
      series: [
        {
          type: "bar",
          barWidth: 18,
          data: rows.map((row) => ({
            value: Number(row.delta_vs_best),
            row,
            itemStyle: {
              color: Number(row.delta_vs_best) <= 0 ? COLORS.accent : COLORS.rose,
            },
          })),
        },
      ],
    });
  }

  function tabarenaTableColumns(mode) {
    if (mode === "overall" || mode === "binary" || mode === "multiclass") {
      return [
        { title: "Method", field: "method", minWidth: 220, formatter: textCellFormatter },
        { title: "Elo", field: "elo", hozAlign: "right", formatter: percentCellFormatter(1) },
        { title: "Rank", field: "rank", hozAlign: "right", formatter: percentCellFormatter(2) },
        { title: "Winrate", field: "winrate", hozAlign: "right", formatter: percentCellFormatter(3) },
        { title: "MRR", field: "mrr", hozAlign: "right", formatter: percentCellFormatter(3) },
        { title: "Metric error", field: "metric_error", hozAlign: "right", formatter: percentCellFormatter(4) },
        { title: "Normalized error", field: "normalized-error", hozAlign: "right", formatter: percentCellFormatter(4) },
      ];
    }
    return [
      { title: "Dataset", field: "dataset", minWidth: 220, formatter: textCellFormatter },
      { title: "Task", field: "problem_type", minWidth: 120, formatter: textCellFormatter },
      { title: "Bal. Acc.", field: "balanced_accuracy", hozAlign: "right", formatter: percentCellFormatter(3) },
      { title: "Metric error", field: "metric_error", hozAlign: "right", formatter: percentCellFormatter(4) },
      { title: "Best official", field: "ref_best_method", minWidth: 180, formatter: textCellFormatter },
      { title: "Delta vs best", field: "delta_vs_best", hozAlign: "right", formatter: percentCellFormatter(4) },
      { title: "Rank", field: "dataset_rank", hozAlign: "right", formatter: intCellFormatter },
      { title: "Selected model", field: "model", minWidth: 160, formatter: textCellFormatter },
    ];
  }

  function ensureTabarenaTable() {
    if (state.tables.tabarena) {
      return state.tables.tabarena;
    }
    const table = new Tabulator(byId("tn-tabarena-table"), {
      layout: "fitColumns",
      pagination: true,
      paginationSize: 12,
      movableColumns: true,
      placeholder: "No TabArena rows match the current filters.",
    });
    state.tables.tabarena = table;
    return table;
  }

  function renderTabarenaTable(filtered) {
    const mode = state.tabarenaFilters.tableMode;
    const table = ensureTabarenaTable();
    const dataMap = {
      per_dataset: filtered.perDataset,
      overall: filtered.leaderboardOverall,
      binary: filtered.leaderboardBinary,
      multiclass: filtered.leaderboardMulticlass,
    };
    table.setColumns(tabarenaTableColumns(mode));
    table.setPageSize(mode === "per_dataset" ? 12 : 16);
    table.replaceData(dataMap[mode] || []);
  }

  function renderTabarena() {
    const filtered = filterTabarenaData();
    renderTabarenaMetrics(filtered);
    renderTabarenaLeaderboardChart(filtered);
    renderTabarenaGapChart(filtered);
    renderTabarenaTable(filtered);
  }

  function setActiveTab(tab) {
    state.activeTab = tab;
    document.querySelectorAll(".tn-browser-tab").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.tab === tab);
    });
    document.querySelectorAll(".tn-browser-panel").forEach((panel) => {
      panel.classList.toggle("is-active", panel.dataset.panel === tab);
    });
    if (tab === "benchmark") {
      renderBenchmark();
      setStatus("HDLSS benchmark browser ready.", "ready");
      return;
    }
    if (!state.tabarenaLoaded) {
      loadTabarena();
      return;
    }
    renderTabarena();
    setStatus("TabArena snapshot ready.", "ready");
  }

  async function loadBenchmark() {
    const url = state.root.dataset.benchmarkUrl;
    const payload = await fetchJson(url);
    state.benchmark = inflateBenchmarkPayload(payload);
    if (!state.benchmark.metadata.available) {
      throw new Error(state.benchmark.metadata.message || "Benchmark payload reported no data.");
    }
    initializeBenchmarkFilters();
    renderBenchmark();
    setStatus("HDLSS benchmark browser ready. TabArena loads on demand.", "ready");
  }

  async function loadTabarena() {
    try {
      setStatus("Loading TabArena snapshot…");
      const payload = await fetchJson(state.root.dataset.tabarenaUrl);
      state.tabarena = inflateTabarenaPayload(payload);
      state.tabarenaLoaded = true;
      if (!state.tabarena.metadata.available) {
        throw new Error(state.tabarena.metadata.message || "TabArena payload reported no data.");
      }
      renderTabarena();
      setStatus("TabArena snapshot ready.", "ready");
    } catch (error) {
      setStatus(error.message, "error");
      const chart = ensureChart("tabarenaLeaderboard", "tn-tabarena-leaderboard-chart");
      if (chart) {
        renderEmptyChart(chart, "TabArena data unavailable.");
      }
      const gapChart = ensureChart("tabarenaGap", "tn-tabarena-gap-chart");
      if (gapChart) {
        renderEmptyChart(gapChart, "TabArena data unavailable.");
      }
    }
  }

  function bindBenchmarkControls() {
    [
      ["tn-benchmark-family", "family"],
      ["tn-benchmark-campaign", "campaign"],
      ["tn-benchmark-tier", "tier"],
      ["tn-benchmark-domain", "domain"],
      ["tn-benchmark-platform", "platform"],
    ].forEach(([id, key]) => {
      const node = byId(id);
      if (!node) {
        return;
      }
      node.addEventListener("change", () => {
        state.benchmarkFilters[key] = node.value;
        renderBenchmark();
      });
    });

    [
      ["tn-benchmark-dataset-query", "datasetQuery"],
      ["tn-benchmark-profile-query", "profileQuery"],
    ].forEach(([id, key]) => {
      const node = byId(id);
      if (!node) {
        return;
      }
      node.addEventListener("input", () => {
        state.benchmarkFilters[key] = node.value;
        renderBenchmark();
      });
    });

    const reset = byId("tn-benchmark-reset");
    if (reset) {
      reset.addEventListener("click", () => {
        state.benchmarkFilters = {
          family: "all",
          campaign: "all",
          tier: "all",
          domain: "all",
          platform: "all",
          datasetQuery: "",
          profileQuery: "",
          selectedDataset: "",
          tableMode: state.benchmarkFilters.tableMode,
        };
        [
          "tn-benchmark-family",
          "tn-benchmark-campaign",
          "tn-benchmark-tier",
          "tn-benchmark-domain",
          "tn-benchmark-platform",
        ].forEach((id) => {
          if (byId(id)) {
            byId(id).value = "all";
          }
        });
        ["tn-benchmark-dataset-query", "tn-benchmark-profile-query"].forEach((id) => {
          if (byId(id)) {
            byId(id).value = "";
          }
        });
        renderBenchmark();
      });
    }

    document.querySelectorAll("[data-table-mode]").forEach((button) => {
      button.addEventListener("click", () => {
        state.benchmarkFilters.tableMode = button.dataset.tableMode;
        document.querySelectorAll("[data-table-mode]").forEach((candidate) => {
          candidate.classList.toggle("is-active", candidate === button);
        });
        renderBenchmark();
      });
    });
  }

  function bindTabarenaControls() {
    const typeNode = byId("tn-tabarena-problem-type");
    if (typeNode) {
      typeNode.addEventListener("change", () => {
        state.tabarenaFilters.problemType = typeNode.value;
        renderTabarena();
      });
    }

    const queryNode = byId("tn-tabarena-query");
    if (queryNode) {
      queryNode.addEventListener("input", () => {
        state.tabarenaFilters.query = queryNode.value;
        renderTabarena();
      });
    }

    const reset = byId("tn-tabarena-reset");
    if (reset) {
      reset.addEventListener("click", () => {
        state.tabarenaFilters.problemType = "all";
        state.tabarenaFilters.query = "";
        if (typeNode) {
          typeNode.value = "all";
        }
        if (queryNode) {
          queryNode.value = "";
        }
        renderTabarena();
      });
    }

    document.querySelectorAll("[data-tabarena-table-mode]").forEach((button) => {
      button.addEventListener("click", () => {
        state.tabarenaFilters.tableMode = button.dataset.tabarenaTableMode;
        document.querySelectorAll("[data-tabarena-table-mode]").forEach((candidate) => {
          candidate.classList.toggle("is-active", candidate === button);
        });
        renderTabarena();
      });
    });
  }

  function bindTabSwitches() {
    document.querySelectorAll(".tn-browser-tab").forEach((button) => {
      button.addEventListener("click", () => setActiveTab(button.dataset.tab));
    });
  }

  async function init() {
    state.root = byId("tn-results-browser");
    if (!state.root) {
      return;
    }
    ensureTabulatorStyles();
    bindTabSwitches();
    bindBenchmarkControls();
    bindTabarenaControls();
    try {
      await loadBenchmark();
    } catch (error) {
      setStatus(error.message, "error");
      const chart = ensureChart("benchmarkDataset", "tn-benchmark-dataset-chart");
      if (chart) {
        renderEmptyChart(chart, "Benchmark data unavailable.");
      }
      const familyChart = ensureChart("benchmarkFamily", "tn-benchmark-family-chart");
      if (familyChart) {
        renderEmptyChart(familyChart, "Benchmark data unavailable.");
      }
      const detailChart = ensureChart("benchmarkDetail", "tn-benchmark-detail-chart");
      if (detailChart) {
        renderEmptyChart(detailChart, "Benchmark data unavailable.");
      }
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
