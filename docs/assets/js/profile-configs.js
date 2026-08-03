(function () {
  "use strict";

  const numberFormatter = new Intl.NumberFormat("en-US");
  const fieldLabelOverrides = {
    df_stage_position: "DF stage position",
    df_stage_source_space: "DF source space",
    fs_ipss_use_eats_threshold: "IPSS EATS threshold",
    fs_stability_target_pfer: "Stability target PFER",
    fs_cap_applied: "Feature cap applied",
    fs_cap_max_allowed: "Feature cap max allowed",
    n_low_gof_downweighted: "Low-GOF downweight count",
    n_dist_skipped_unreliable: "Skipped unreliable distributions",
    classifier_conformal_mapie_enabled: "MAPIE enabled",
    classifier_conformal_mapie_method: "MAPIE method",
    classifier_conformal_applied: "Conformal applied",
    classifier_conformal_enabled: "Conformal enabled",
  };

  document.addEventListener("DOMContentLoaded", () => {
    const root = document.getElementById("tn-profile-config-browser");
    if (!root) {
      return;
    }

    const elements = {
      status: document.getElementById("tn-profile-config-status"),
      query: document.getElementById("tn-profile-config-query"),
      select: document.getElementById("tn-profile-config-select"),
      metrics: document.getElementById("tn-profile-config-metrics"),
      fixed: document.getElementById("tn-profile-config-fixed"),
      varying: document.getElementById("tn-profile-config-varying"),
      seeds: document.getElementById("tn-profile-config-seeds"),
      models: document.getElementById("tn-profile-config-models"),
      datasets: document.getElementById("tn-profile-config-datasets"),
    };

    if (Object.values(elements).some((node) => !node)) {
      return;
    }

    const state = {
      profiles: [],
      filteredProfiles: [],
      query: "",
      selectedProfile: "",
    };

    const requestedProfile = new URL(window.location.href).searchParams.get("profile") || "";
    const dataUrl = root.dataset.profileCatalogUrl;

    elements.query.addEventListener("input", () => {
      state.query = elements.query.value.trim().toLowerCase();
      syncFilteredProfiles();
      render();
    });

    elements.select.addEventListener("change", () => {
      state.selectedProfile = elements.select.value;
      render();
    });

    loadData(dataUrl)
      .then((payload) => {
        const profiles = Array.isArray(payload && payload.profiles) ? payload.profiles.slice() : [];
        profiles.sort((left, right) => {
          const familyCompare = String(left.family_label || left.family || "").localeCompare(
            String(right.family_label || right.family || ""),
          );
          if (familyCompare !== 0) {
            return familyCompare;
          }
          return String(left.profile || "").localeCompare(String(right.profile || ""));
        });
        state.profiles = profiles;
        state.selectedProfile = requestedProfile;
        syncFilteredProfiles();
        setStatus(
          `Ready: ${formatNumber(profiles.length)} profiles across ${formatNumber(
            payload.metadata && payload.metadata.run_count,
          )} published run rows.`,
          "ready",
        );
        render();
      })
      .catch((error) => {
        console.error(error);
        setStatus("Unable to load the published profile configuration bundle.", "error");
      });

    function syncFilteredProfiles() {
      const query = state.query;
      state.filteredProfiles = state.profiles.filter((profile) => {
        if (!query) {
          return true;
        }
        const haystack = [
          profile.profile,
          profile.family,
          profile.family_label,
          profile.campaign,
          profile.campaign_label,
        ]
          .filter(Boolean)
          .join(" ")
          .toLowerCase();
        return haystack.includes(query);
      });

      const availableNames = new Set(state.filteredProfiles.map((profile) => String(profile.profile || "")));
      if (!availableNames.has(state.selectedProfile)) {
        state.selectedProfile = state.filteredProfiles.length ? String(state.filteredProfiles[0].profile || "") : "";
      }
      populateSelect();
    }

    function populateSelect() {
      const options = state.filteredProfiles.map((profile) => {
        const name = String(profile.profile || "");
        const campaignLabel = profile.campaign_label || profile.campaign || "Unknown campaign";
        return `<option value="${escapeHtml(name)}"${name === state.selectedProfile ? " selected" : ""}>${escapeHtml(
          `${name} · ${campaignLabel}`,
        )}</option>`;
      });
      elements.select.innerHTML = options.length
        ? options.join("")
        : '<option value="">No matching profiles</option>';
    }

    function render() {
      const profile = state.filteredProfiles.find((item) => String(item.profile || "") === state.selectedProfile);
      if (!profile) {
        clearPanels();
        const suffix = state.query ? ` for "${state.query}"` : "";
        setStatus(`No profiles matched${suffix}.`, "error");
        updateUrl("");
        return;
      }

      setStatus(
        `Showing ${escapeText(profile.profile)} with ${formatNumber(profile.run_count)} run rows across ${formatNumber(
          profile.dataset_count,
        )} datasets.`,
        "ready",
      );
      updateUrl(String(profile.profile || ""));
      renderMetrics(profile);
      elements.fixed.textContent = JSON.stringify(profile.fixed_parameters || {}, null, 2);
      renderVarying(profile);
      renderTable(
        elements.seeds,
        ["Seed", "Run rows"],
        Array.isArray(profile.seed_run_counts)
          ? profile.seed_run_counts.map((item) => [String(item.seed), formatNumber(item.run_count)])
          : [],
        "No seed rows were available for this profile.",
      );
      renderTable(
        elements.models,
        ["Model", "Run rows"],
        Array.isArray(profile.models)
          ? profile.models.map((item) => [`<code>${escapeHtml(String(item.model || "unknown"))}</code>`, formatNumber(item.run_count)])
          : [],
        "No model rows were available for this profile.",
      );
      renderTable(
        elements.datasets,
        ["Dataset", "Dataset ID", "Run rows"],
        Array.isArray(profile.datasets)
          ? profile.datasets.map((item) => [
              escapeHtml(String(item.dataset_label || item.dataset_id || "Unknown dataset")),
              `<code>${escapeHtml(String(item.dataset_id || "unknown"))}</code>`,
              formatNumber(item.run_count),
            ])
          : [],
        "No dataset rows were available for this profile.",
      );
    }

    function renderMetrics(profile) {
      const seeds = Array.isArray(profile.seeds) ? profile.seeds : [];
      const models = Array.isArray(profile.models) ? profile.models : [];
      const cards = [
        metricCard("Profile", escapeText(profile.profile), escapeText(profile.family_label || profile.family || "Unknown family")),
        metricCard("Campaign", escapeText(profile.campaign_label || profile.campaign || "Unknown campaign"), escapeText(profile.campaign || "")),
        metricCard("Scope", escapeText(profile.campaign_scope_label || profile.campaign_scope || "Unknown scope"), ""),
        metricCard("Run rows", formatNumber(profile.run_count), `${formatNumber(profile.dataset_count)} datasets`),
        metricCard("Seeds", formatNumber(seeds.length), seeds.length ? seeds.map((value) => escapeText(String(value))).join(", ") : "None"),
        metricCard("Models", formatNumber(models.length), models.length ? "Observed final picks in the published rows" : "None"),
      ];
      elements.metrics.innerHTML = cards.join("");
    }

    function renderVarying(profile) {
      const varying = Array.isArray(profile.varying_parameters) ? profile.varying_parameters : [];
      if (!varying.length) {
        elements.varying.innerHTML = '<p class="tn-browser-empty">No varying settings were recorded for this profile.</p>';
        return;
      }
      const rows = varying.map((item) => {
        const values = Array.isArray(item.values) ? item.values : [];
        const valueHtml = values
          .map((entry) => `<code>${escapeHtml(formatValue(entry.value))}</code> (${formatNumber(entry.run_count)} runs)`)
          .join("<br>");
        return [
          escapeHtml(humanizeField(String(item.key || ""))),
          `${valueHtml}${item.distinct_count > values.length ? `<br>and ${formatNumber(item.distinct_count - values.length)} more values` : ""}`,
        ];
      });
      renderTable(elements.varying, ["Field", "Observed values"], rows, "No varying settings were recorded for this profile.");
    }

    function renderTable(container, headers, rows, emptyMessage) {
      if (!rows.length) {
        container.innerHTML = `<p class="tn-browser-empty">${escapeHtml(emptyMessage)}</p>`;
        return;
      }
      const head = headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("");
      const body = rows
        .map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`)
        .join("");
      container.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
    }

    function clearPanels() {
      elements.metrics.innerHTML = "";
      elements.fixed.textContent = "{}";
      elements.varying.innerHTML = "";
      elements.seeds.innerHTML = "";
      elements.models.innerHTML = "";
      elements.datasets.innerHTML = "";
    }

    function setStatus(message, kind) {
      elements.status.classList.remove("is-ready", "is-error");
      if (kind === "ready") {
        elements.status.classList.add("is-ready");
      } else if (kind === "error") {
        elements.status.classList.add("is-error");
      }
      elements.status.textContent = message;
    }

    function updateUrl(profileName) {
      const url = new URL(window.location.href);
      if (profileName) {
        url.searchParams.set("profile", profileName);
      } else {
        url.searchParams.delete("profile");
      }
      window.history.replaceState({}, "", url.toString());
    }
  });

  async function loadData(dataUrl) {
    const response = await fetch(dataUrl, { credentials: "same-origin" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return response.json();
  }

  function metricCard(label, value, sub) {
    return `
      <div class="tn-metric-card">
        <div class="tn-metric-label">${escapeHtml(label)}</div>
        <div class="tn-metric-value">${escapeHtml(value)}</div>
        <div class="tn-metric-sub">${escapeHtml(sub)}</div>
      </div>
    `;
  }

  function humanizeField(key) {
    if (fieldLabelOverrides[key]) {
      return fieldLabelOverrides[key];
    }
    return key
      .replace(/_/g, " ")
      .replace(/\bdf\b/gi, "DF")
      .replace(/\bfs\b/gi, "FS")
      .replace(/\bipss\b/gi, "IPSS")
      .replace(/\bpfer\b/gi, "PFER")
      .replace(/\bmapie\b/gi, "MAPIE")
      .replace(/\bmnpo\b/gi, "MNPO")
      .replace(/\bgof\b/gi, "GOF")
      .replace(/\b\w/g, (match) => match.toUpperCase());
  }

  function formatValue(value) {
    if (Array.isArray(value)) {
      return value.length ? value.map((item) => formatValue(item)).join(", ") : "[]";
    }
    if (value && typeof value === "object") {
      return JSON.stringify(value);
    }
    if (typeof value === "boolean") {
      return value ? "true" : "false";
    }
    return String(value);
  }

  function formatNumber(value) {
    return numberFormatter.format(Number(value || 0));
  }

  function escapeText(value) {
    return String(value || "");
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }
})();
