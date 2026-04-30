const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000/api";

const STORAGE_KEYS = {
  wardrobeImageCache: "ai-wardrobe-image-cache-v1",
  wardrobeIndexCache: "ai-wardrobe-item-index-v1",
};

const COLOR_HEX_BY_NAME = {
  black: "#111111",
  gray: "#6b7280",
  grey: "#6b7280",
  white: "#f3f4f6",
  blue: "#3b82f6",
  purple: "#8b5cf6",
  green: "#22c55e",
  red: "#ef4444",
  orange: "#f97316",
  yellow: "#facc15",
  pink: "#ec4899",
  brown: "#8b5e3c",
  beige: "#d6c0a3",
  navy: "#1e3a8a",
};

function canUseStorage() {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

function readStorageJson(key, fallback) {
  if (!canUseStorage()) return fallback;

  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw);
  } catch {
    return fallback;
  }
}

function writeStorageJson(key, value) {
  if (!canUseStorage()) return;

  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Ignore storage quota and serialization errors gracefully.
  }
}

function normalizeToken(value) {
  return String(value || "").trim().toLowerCase();
}

function readWardrobeImageCache() {
  const parsed = readStorageJson(STORAGE_KEYS.wardrobeImageCache, {});
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return {};
  }
  return parsed;
}

function writeWardrobeImageCache(cache) {
  const payload = cache && typeof cache === "object" ? cache : {};
  writeStorageJson(STORAGE_KEYS.wardrobeImageCache, payload);
}

function readWardrobeIndexCache() {
  const parsed = readStorageJson(STORAGE_KEYS.wardrobeIndexCache, []);
  return Array.isArray(parsed) ? parsed : [];
}

function writeWardrobeIndexCache(entries) {
  const payload = Array.isArray(entries) ? entries.slice(0, 300) : [];
  writeStorageJson(STORAGE_KEYS.wardrobeIndexCache, payload);
}

function upsertWardrobeIndexEntry(entry) {
  const itemId = String(entry?.item_id || "").trim();
  if (!itemId) return;

  const list = readWardrobeIndexCache();
  const existingIndex = list.findIndex((row) => String(row?.item_id || "") === itemId);
  const existing = existingIndex >= 0 ? list[existingIndex] : null;

  const nextRow = {
    item_id: itemId,
    category: String(entry?.category || existing?.category || "Unknown"),
    color: String(entry?.color || existing?.color || "gray"),
    image_data_url:
      typeof entry?.image_data_url === "string" && entry.image_data_url.startsWith("data:image/")
        ? entry.image_data_url
        : existing?.image_data_url || null,
    updated_at: Date.now(),
  };

  if (existingIndex >= 0) {
    list.splice(existingIndex, 1);
  }
  list.unshift(nextRow);
  writeWardrobeIndexCache(list);
}

function cacheWardrobeImage(itemId, imageDataUrl, category, color) {
  const normalizedItemId = String(itemId || "").trim();
  if (!normalizedItemId) return;

  let cachedImage = null;
  if (typeof imageDataUrl === "string" && imageDataUrl.startsWith("data:image/")) {
    const cache = readWardrobeImageCache();
    cache[normalizedItemId] = imageDataUrl;
    writeWardrobeImageCache(cache);
    cachedImage = imageDataUrl;
  } else {
    const cache = readWardrobeImageCache();
    cachedImage = typeof cache[normalizedItemId] === "string" ? cache[normalizedItemId] : null;
  }

  upsertWardrobeIndexEntry({
    item_id: normalizedItemId,
    category,
    color,
    image_data_url: cachedImage,
  });
}

function removeCachedWardrobeItem(itemId) {
  const normalizedItemId = String(itemId || "").trim();
  if (!normalizedItemId) return;

  const imageCache = readWardrobeImageCache();
  if (Object.prototype.hasOwnProperty.call(imageCache, normalizedItemId)) {
    delete imageCache[normalizedItemId];
    writeWardrobeImageCache(imageCache);
  }

  const nextIndex = readWardrobeIndexCache().filter(
    (row) => String(row?.item_id || "") !== normalizedItemId
  );
  writeWardrobeIndexCache(nextIndex);
}

function resolveCachedImage(item, imageCache, indexCache) {
  const direct =
    typeof item?.image_data_url === "string" && item.image_data_url.startsWith("data:image/")
      ? item.image_data_url
      : null;
  if (direct) {
    return direct;
  }

  const idCandidates = [item?.item_id, item?.wardrobe_item_id, item?.id]
    .filter(Boolean)
    .map((value) => String(value));

  for (const candidate of idCandidates) {
    if (typeof imageCache[candidate] === "string") {
      return imageCache[candidate];
    }
  }

  for (const candidate of idCandidates) {
    const exact = indexCache.find((row) => String(row?.item_id || "") === candidate);
    if (typeof exact?.image_data_url === "string") {
      return exact.image_data_url;
    }
  }

  const category = normalizeToken(item?.category);
  const color = normalizeToken(item?.color);
  if (!category) {
    return null;
  }

  const match = indexCache.find((row) => {
    if (normalizeToken(row?.category) !== category) {
      return false;
    }
    if (color && normalizeToken(row?.color) !== color) {
      return false;
    }
    return typeof row?.image_data_url === "string";
  });

  return match?.image_data_url || null;
}

function clampChannel(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return Math.max(0, Math.min(255, Math.round(numeric)));
}

function rgbToHex(rgb) {
  if (!Array.isArray(rgb) || rgb.length < 3) return null;
  const [r, g, b] = rgb;
  const toHex = (v) => clampChannel(v).toString(16).padStart(2, "0");
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function colorNameToHex(name) {
  if (!name) return COLOR_HEX_BY_NAME.gray;
  return COLOR_HEX_BY_NAME[String(name).toLowerCase()] || COLOR_HEX_BY_NAME.gray;
}

function normalizeColorEntry(color) {
  if (!color) return null;
  const name = color.name || color.color || "gray";
  const hex = color.hex || rgbToHex(color.rgb) || colorNameToHex(name);
  return { name, hex };
}

function normalizeClassifyResult(data) {
  const normalizedColors = Array.isArray(data?.colors)
    ? data.colors.map(normalizeColorEntry).filter(Boolean)
    : [];

  const fallbackColorName =
    data?.color_name || data?.color || normalizedColors[0]?.name || "gray";

  const colors = normalizedColors.length
    ? normalizedColors
    : [{ name: fallbackColorName, hex: colorNameToHex(fallbackColorName) }];

  return {
    ...data,
    color_name: fallbackColorName,
    colors,
    features: Array.isArray(data?.features) ? data.features : [],
  };
}

function mapOccasion(occasion) {
  const mapping = {
    casual: "casual",
    formal: "formal",
    athletic: "sport",
    business: "formal",
    sport: "sport",
  };
  return mapping[occasion] || "casual";
}

function mapWeather(weather) {
  const mapping = {
    clear: "mild",
    cloudy: "mild",
    rain: "rainy",
    rainy: "rainy",
    snow: "cold",
    cold: "cold",
    hot: "hot",
    mild: "mild",
  };
  return mapping[weather] || "mild";
}

function mapGoalMode(goalMode) {
  const mapping = {
    balanced: "balanced",
    formal_precision: "formal_precision",
    comfort_first: "comfort_first",
    heat_survival: "heat_survival",
    rain_safe: "rain_safe",
    repeat_avoider: "repeat_avoider",
    bold_experiment: "bold_experiment",
  };
  return mapping[goalMode] || "balanced";
}

function mapColorStrategy(strategy) {
  const mapping = {
    auto: "auto",
    monochrome: "monochrome",
    analogous: "analogous",
    complementary: "complementary",
    neutral_plus_one: "neutral_plus_one",
    high_contrast: "high_contrast",
  };
  return mapping[strategy] || "auto";
}

function mapTemperatureBias(temperatureBias) {
  const mapping = {
    neutral: "neutral",
    run_cold: "run_cold",
    run_warm: "run_warm",
  };
  return mapping[temperatureBias] || "neutral";
}

function mapExplainability(explainability) {
  const mapping = {
    rule: "rule",
    llm: "llm",
  };
  return mapping[explainability] || "rule";
}

function normalizeOutfitCollection(outfits, imageCache, indexCache) {
  return Array.isArray(outfits)
    ? outfits.map((outfit, outfitIndex) => ({
        ...outfit,
        items: Array.isArray(outfit?.items)
          ? outfit.items.map((item, itemIndex) => {
              const colorName = item?.color || "gray";
              const image_data_url = resolveCachedImage(item, imageCache, indexCache);
              return {
                ...item,
                id: item?.id || `${outfitIndex}-${itemIndex}`,
                image_data_url,
                colors: [{ name: colorName, hex: colorNameToHex(colorName) }],
              };
            })
          : [],
      }))
    : [];
}

async function fetchWithRetry(url, options = {}, retries = 2) {
  const maxRetries = Math.max(0, Number(retries) || 0);

  for (let i = 0; i <= maxRetries; i++) {
    try {
      const res = await fetch(url, options);
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        const error = new Error(errorData.detail || `HTTP error! status: ${res.status}`);
        error.status = res.status;

        // Retry only for server errors. Client-side 4xx should fail fast.
        if (res.status < 500 || i === maxRetries) {
          throw error;
        }

        await new Promise((resolve) => setTimeout(resolve, 500 * (i + 1)));
        continue;
      }
      return await res.json();
    } catch (err) {
      if (i === maxRetries) throw err;

      // Errors with status are HTTP errors already handled above.
      if (typeof err?.status === "number") {
        throw err;
      }

      await new Promise((resolve) => setTimeout(resolve, 500 * (i + 1)));
    }
  }
}

export const api = {
  checkHealth: () => fetchWithRetry(`${API_BASE}/health`),
  
  getCategories: () => fetchWithRetry(`${API_BASE}/categories`),
  
  getSamples: async () => {
    const payload = await fetchWithRetry(`${API_BASE}/samples`);
    const samples = Array.isArray(payload?.samples) ? payload.samples : [];

    return samples
      .map((sample, index) => ({
        ...sample,
        id: sample.id || `sample-${index}`,
        image_base64: sample.image_base64 || sample.base64 || "",
      }))
      .filter((sample) => sample.image_base64);
  },
  
  classifyImage: async (file) => {
    const formData = new FormData();
    formData.append("file", file);
    const response = await fetchWithRetry(`${API_BASE}/classify`, {
      method: 'POST',
      body: formData,
    });
    return normalizeClassifyResult(response);
  },

  classifySample: async (base64) => {
    const response = await fetchWithRetry(`${API_BASE}/classify-sample`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base64 }),
    });
    return normalizeClassifyResult(response);
  },

  getWardrobe: async () => {
    const payload = await fetchWithRetry(`${API_BASE}/wardrobe`);
    const items = Array.isArray(payload?.items) ? payload.items : [];
    const imageCache = readWardrobeImageCache();
    const indexCache = readWardrobeIndexCache();

    return items.map((item, index) => {
      const id = item.item_id || item.id || `item-${index}`;
      const colorName =
        item.color || item.color_name || item.colors?.[0]?.name || "gray";
      const image_data_url = resolveCachedImage({ ...item, item_id: id }, imageCache, indexCache);

      upsertWardrobeIndexEntry({
        item_id: id,
        category: item?.category || "Unknown",
        color: colorName,
        image_data_url,
      });

      return {
        ...item,
        id,
        item_id: id,
        color: colorName,
        confidence: Number(item?.confidence ?? 0),
        label: item?.label || item?.category || "Unknown",
        created_at: item?.created_at || null,
        features: Array.isArray(item?.features) ? item.features : [],
        image_data_url,
        colors: [{ name: colorName, hex: colorNameToHex(colorName) }],
      };
    });
  },
  
  addToWardrobe: async (item) => {
    const generatedId =
      typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
        ? crypto.randomUUID()
        : `item-${Date.now()}`;

    const payload = {
      item_id: item?.item_id || item?.id || generatedId,
      category: item?.category || "Unknown",
      color: item?.color_name || item?.color || item?.colors?.[0]?.name || "gray",
      confidence: Number(item?.confidence ?? 0),
      label: item?.label || item?.category || "Unknown",
      features: Array.isArray(item?.features) ? item.features : [],
    };

    const response = await fetchWithRetry(`${API_BASE}/wardrobe/add`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    cacheWardrobeImage(
      payload.item_id,
      item?.image_data_url,
      payload.category,
      payload.color
    );

    return response;
  },

  removeFromWardrobe: async (id) => {
    const res = await fetch(`${API_BASE}/wardrobe/${id}`, {
      method: "DELETE"
    });
    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${res.status}`);
    }

    removeCachedWardrobeItem(id);
    return true; // 204 No Content expected
  },

  recommendOutfits: async (params) => {
    const payload = {
      occasion: mapOccasion(params?.occasion),
      weather: mapWeather(params?.weather),
      top_k: Number(params?.top_k || 3),
      goal_mode: mapGoalMode(params?.goal_mode),
      exploration: Number(params?.exploration ?? 0.35),
      color_strategy: mapColorStrategy(params?.color_strategy),
      occasion_strictness: Number(params?.occasion_strictness ?? 0.6),
      anti_repeat: params?.anti_repeat !== false,
      hero_item_id: params?.hero_item_id || undefined,
      temperature_bias: mapTemperatureBias(params?.temperature_bias),
      explainability: mapExplainability(params?.explainability),
      include_backup_pack: params?.include_backup_pack !== false,
      feedback_learning: params?.feedback_learning !== false,
    };

    const response = await fetchWithRetry(`${API_BASE}/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const imageCache = readWardrobeImageCache();
    const indexCache = readWardrobeIndexCache();

    const outfits = normalizeOutfitCollection(response?.outfits, imageCache, indexCache);
    const backupEntries = response?.backup_pack && typeof response.backup_pack === "object"
      ? Object.entries(response.backup_pack)
      : [];

    const backupPack = backupEntries.reduce((accumulator, [label, outfit]) => {
      const normalized = normalizeOutfitCollection([outfit], imageCache, indexCache);
      if (normalized[0]) {
        accumulator[label] = normalized[0];
      }
      return accumulator;
    }, {});

    return {
      ...response,
      outfits,
      backup_pack: backupPack,
      closet_gap_insights: Array.isArray(response?.closet_gap_insights)
        ? response.closet_gap_insights
        : [],
    };
  },

  submitRecommendationFeedback: async (payload) => {
    return fetchWithRetry(`${API_BASE}/recommend/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        recommendation_id: String(payload?.recommendation_id || ''),
        signal: payload?.signal === 'dislike' ? 'dislike' : 'like',
      }),
    });
  },

  getRecommendationFeedbackProfile: async () => {
    return fetchWithRetry(`${API_BASE}/recommend/feedback-profile`);
  }
};
