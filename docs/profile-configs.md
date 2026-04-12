---
title: Profile Config Browser
nav_order: 8
---

# Profile Config Browser

Use this page when you want to inspect the published run setup behind a benchmark profile: the settings that stayed fixed, the settings that changed across datasets or routing branches, and the exact seed set carried by the public bundle.

<div
  id="tn-profile-config-browser"
  class="tn-results-browser"
  data-profile-catalog-url="{{ '/assets/data/benchmark_profile_catalog.json' | relative_url }}"
>
  <div class="tn-browser-shell">
    <section class="tn-browser-hero">
      <div>
        <p class="tn-browser-kicker">Profile configs</p>
        <h2>Published run settings</h2>
        <p class="tn-browser-copy">
          Pick a profile to see the fixed pipeline settings, the observed run-time differences inside that profile, and the seed set behind the published benchmark rows.
        </p>
      </div>
      <div class="tn-browser-links">
        <a class="tn-browser-link" href="{{ '/results-browser.html' | relative_url }}">Results browser</a>
        <a class="tn-browser-link" href="{{ '/assets/data/benchmark_profile_catalog.json' | relative_url }}">Profile config JSON</a>
      </div>
    </section>

    <section class="tn-browser-status" id="tn-profile-config-status">
      Loading profile configuration data…
    </section>

    <section class="tn-browser-controls">
      <label class="tn-browser-grow">
        Profile search
        <input id="tn-profile-config-query" type="search" placeholder="e.g. N04, V20_F, C_ONLY" />
      </label>
      <label class="tn-browser-grow">
        Profile
        <select id="tn-profile-config-select">
          <option value="">Loading profiles…</option>
        </select>
      </label>
    </section>

    <div class="tn-browser-metrics" id="tn-profile-config-metrics"></div>

    <div class="tn-browser-grid">
      <article class="tn-browser-card">
        <div class="tn-browser-card-head">
          <h3>Fixed settings</h3>
          <p>These values stayed constant across every published row for the selected profile.</p>
        </div>
        <pre class="tn-browser-json" id="tn-profile-config-fixed"></pre>
      </article>

      <article class="tn-browser-card">
        <div class="tn-browser-card-head">
          <h3>Observed variations</h3>
          <p>These fields changed across datasets or run-time routing inside the same profile.</p>
        </div>
        <div id="tn-profile-config-varying"></div>
      </article>

      <article class="tn-browser-card tn-browser-span-2">
        <div class="tn-browser-card-head">
          <h3>Seeds, models, and dataset coverage</h3>
          <p>The public bundle keeps the seed set, the observed model picks, and the dataset coverage for each profile.</p>
        </div>
        <div class="tn-browser-grid">
          <div>
            <h4>Seeds</h4>
            <div id="tn-profile-config-seeds"></div>
          </div>
          <div>
            <h4>Models</h4>
            <div id="tn-profile-config-models"></div>
          </div>
          <div class="tn-browser-span-2">
            <h4>Datasets</h4>
            <div id="tn-profile-config-datasets"></div>
          </div>
        </div>
      </article>
    </div>
  </div>
</div>

<noscript>
  <p>This page needs JavaScript enabled to render the profile configuration browser.</p>
</noscript>

<script src="{{ '/assets/js/profile-configs.js' | relative_url }}"></script>

---

> Documentation and webpages on this site are generated from authoritative internal sources using a combination of deterministic rules and generative AI. Errors are possible. Please report issues via [GitHub Discussions](https://github.com/klokedm/tabnetics-public/discussions) or email [marko@tabnetics.org](mailto:marko@tabnetics.org).
