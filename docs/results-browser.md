---
title: Results Browser
nav_order: 5
---

# Results Browser

This browser complements the narrative [Benchmark Results](RESULTS.md) and [TabArena Results](TABARENA_RESULTS.md) pages with an interactive explorer backed by the published public data bundles.

The benchmark tab covers the public HDLSS Val-18 / Val-19 / Val-20 bundle, including per-run metrics, profile summaries, dataset metadata, and the SOTA comparison bands shown in the results page. The TabArena tab loads the public general-tabular snapshot when that bundle is available at publish time.

Use the benchmark tab when you want to slice the HDLSS validation surface by family, campaign, campaign scope, tier, or domain. Use the TabArena tab when you want to inspect the general-tabular comparison snapshot and the per-dataset gap against the current official best method. For a compact guide to the campaign families and datasets exposed here, see [Browser Data Guide](results-browser-data.md). When you want the published seeds and run settings behind a profile, open the [Profile Config Browser](profile-configs.md).

<div
  id="tn-results-browser"
  class="tn-results-browser"
  data-benchmark-url="{{ '/assets/data/benchmark_browser.json' | relative_url }}"
  data-tabarena-url="{{ '/assets/data/tabarena_browser.json' | relative_url }}"
>
  <div class="tn-browser-shell">
    <section class="tn-browser-hero">
      <div>
        <p class="tn-browser-kicker">Interactive browser</p>
        <h2>Interactive result explorer</h2>
        <p class="tn-browser-copy">
          Explore the published benchmark results at your own pace, compare the strongest profiles, and jump into the exact seeds and run settings behind any profile when you want the full picture.
        </p>
      </div>
      <div class="tn-browser-links">
        <a class="tn-browser-link" href="{{ '/assets/data/benchmark_browser.json' | relative_url }}">Benchmark JSON</a>
        <a class="tn-browser-link" href="{{ '/assets/data/tabarena_browser.json' | relative_url }}">TabArena JSON</a>
        <a class="tn-browser-link" href="{{ '/profile-configs.html' | relative_url }}">Profile configs</a>
      </div>
    </section>

    <section class="tn-browser-status" id="tn-browser-status">
      Loading browser data…
    </section>

    <section class="tn-browser-tabs" role="tablist" aria-label="Results browser tabs">
      <button class="tn-browser-tab is-active" type="button" data-tab="benchmark">HDLSS Benchmark</button>
      <button class="tn-browser-tab" type="button" data-tab="tabarena">TabArena</button>
    </section>

    <section class="tn-browser-panel is-active" id="tn-panel-benchmark" data-panel="benchmark">
      <div class="tn-browser-controls">
        <label>
          Family
          <select id="tn-benchmark-family">
            <option value="all">All families</option>
          </select>
        </label>
        <label>
          Campaign
          <select id="tn-benchmark-campaign">
            <option value="all">All campaigns</option>
          </select>
        </label>
        <label>
          Campaign scope
          <select id="tn-benchmark-campaign-scope">
            <option value="all">All campaign scopes</option>
            <option value="full">Full benchmark panel</option>
            <option value="diagnostic">Diagnostic subset</option>
          </select>
        </label>
        <label>
          Tier
          <select id="tn-benchmark-tier">
            <option value="all">All tiers</option>
          </select>
        </label>
        <label>
          Domain
          <select id="tn-benchmark-domain">
            <option value="all">All domains</option>
          </select>
        </label>
        <label>
          Platform
          <select id="tn-benchmark-platform">
            <option value="all">All platforms</option>
          </select>
        </label>
        <label class="tn-browser-grow">
          Dataset search
          <input id="tn-benchmark-dataset-query" type="search" placeholder="e.g. TCGA, Leukemia, CuMiDa" />
        </label>
        <label class="tn-browser-grow">
          Profile search
          <input id="tn-benchmark-profile-query" type="search" placeholder="e.g. N04, C_ONLY, V20_F" />
        </label>
        <button class="tn-browser-ghost" id="tn-benchmark-reset" type="button">Reset filters</button>
      </div>

      <div class="tn-browser-metrics" id="tn-benchmark-metrics"></div>

      <div class="tn-browser-grid">
        <article class="tn-browser-card">
          <div class="tn-browser-card-head">
            <h3>Dataset landscape</h3>
            <p>Best filtered profile per dataset. Click a point to focus the dataset detail view.</p>
          </div>
          <div class="tn-browser-chart" id="tn-benchmark-dataset-chart"></div>
        </article>

        <article class="tn-browser-card">
          <div class="tn-browser-card-head">
            <h3>Family frontier</h3>
            <p>Mean filtered profile performance by experiment family.</p>
          </div>
          <div class="tn-browser-chart" id="tn-benchmark-family-chart"></div>
        </article>

        <article class="tn-browser-card tn-browser-span-2">
          <div class="tn-browser-card-head">
            <h3>Dataset detail</h3>
            <p>Top profiles for the selected dataset, or the global filtered frontier when nothing is selected.</p>
          </div>
          <div class="tn-browser-chart tn-browser-chart-tall" id="tn-benchmark-detail-chart"></div>
        </article>
      </div>

      <div class="tn-browser-table-head">
        <div class="tn-browser-table-modes">
          <button class="tn-browser-mode is-active" type="button" data-table-mode="datasets">Dataset Summary</button>
          <button class="tn-browser-mode" type="button" data-table-mode="profiles">Profile Summary</button>
          <button class="tn-browser-mode" type="button" data-table-mode="runs">Run Explorer</button>
        </div>
        <div class="tn-browser-selection" id="tn-benchmark-selection"></div>
      </div>
      <div class="tn-browser-table" id="tn-benchmark-table"></div>
    </section>

    <section class="tn-browser-panel" id="tn-panel-tabarena" data-panel="tabarena">
      <div class="tn-browser-controls">
        <label>
          Problem type
          <select id="tn-tabarena-problem-type">
            <option value="all">All tasks</option>
            <option value="binary">Binary</option>
            <option value="multiclass">Multiclass</option>
          </select>
        </label>
        <label class="tn-browser-grow">
          Dataset or method search
          <input id="tn-tabarena-query" type="search" placeholder="e.g. APSFailure, RF, TabPFN" />
        </label>
        <button class="tn-browser-ghost" id="tn-tabarena-reset" type="button">Reset filters</button>
      </div>

      <div class="tn-browser-metrics" id="tn-tabarena-metrics"></div>

      <div class="tn-browser-grid">
        <article class="tn-browser-card">
          <div class="tn-browser-card-head">
            <h3>Leaderboard snapshot</h3>
            <p>Overall Elo ladder with the current `tabnetics (general)` row highlighted.</p>
          </div>
          <div class="tn-browser-chart tn-browser-chart-tall" id="tn-tabarena-leaderboard-chart"></div>
        </article>

        <article class="tn-browser-card">
          <div class="tn-browser-card-head">
            <h3>Per-dataset gap to best official method</h3>
            <p>Positive deltas are behind the official best; negative deltas mean tabnetics wins that dataset slice.</p>
          </div>
          <div class="tn-browser-chart tn-browser-chart-tall" id="tn-tabarena-gap-chart"></div>
        </article>
      </div>

      <div class="tn-browser-table-head">
        <div class="tn-browser-table-modes">
          <button class="tn-browser-mode is-active" type="button" data-tabarena-table-mode="per_dataset">Per Dataset</button>
          <button class="tn-browser-mode" type="button" data-tabarena-table-mode="overall">Overall Elo</button>
          <button class="tn-browser-mode" type="button" data-tabarena-table-mode="binary">Binary Elo</button>
          <button class="tn-browser-mode" type="button" data-tabarena-table-mode="multiclass">Multiclass Elo</button>
        </div>
      </div>
      <div class="tn-browser-table" id="tn-tabarena-table"></div>
    </section>
  </div>
</div>

<noscript>
  <p>This page needs JavaScript enabled to render the interactive browser.</p>
</noscript>

<script src="https://cdn.jsdelivr.net/npm/echarts@6.0.0/dist/echarts.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/tabulator-tables@6.3.1/dist/js/tabulator.min.js"></script>
<script src="{{ '/assets/js/results-browser.js' | relative_url }}"></script>

---

> Documentation and webpages on this site are generated from authoritative internal sources using a combination of deterministic rules and generative AI. Errors are possible. Please report issues via [GitHub Discussions](https://github.com/klokedm/tabnetics-public/discussions) or email [marko@tabnetics.org](mailto:marko@tabnetics.org).
