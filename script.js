(function () {
  var root = document.documentElement;
  var toggle = document.querySelector(".theme-toggle");
  var menuLinks = document.querySelectorAll(".main-menu a");
  var sections = document.querySelectorAll(".content-region");
  var storageKey = "preferred-theme";

  root.classList.add("js-enabled");

  function readSavedTheme() {
    try {
      return window.localStorage.getItem(storageKey);
    } catch (error) {
      return null;
    }
  }

  function saveTheme(theme) {
    try {
      window.localStorage.setItem(storageKey, theme);
    } catch (error) {
      return;
    }
  }

  function getDefaultRegion() {
    return menuLinks.length ? menuLinks[0].getAttribute("href") : "";
  }

  function getRegionFromHash() {
    var hash = window.location.hash || getDefaultRegion();
    var sectionId = hash.charAt(0) === "#" ? hash.slice(1) : "";

    if (sectionId && document.getElementById(sectionId)) {
      return hash;
    }

    return getDefaultRegion();
  }

  function showRegion(region) {
    sections.forEach(function (section) {
      section.classList.toggle("active", "#" + section.id === region);
    });

    menuLinks.forEach(function (link) {
      link.classList.toggle("active", link.getAttribute("href") === region);
    });
  }

  function applyTheme(theme) {
    var nextTheme = theme === "light" ? "light" : "dark";
    root.setAttribute("data-theme", nextTheme);

    if (toggle) {
      var isLight = nextTheme === "light";
      toggle.setAttribute("aria-pressed", String(isLight));
      toggle.setAttribute("aria-label", isLight ? "Switch to dark mode" : "Switch to light mode");
    }
  }

  applyTheme(readSavedTheme() || "dark");
  showRegion(getRegionFromHash());

  window.addEventListener("hashchange", function () {
    showRegion(getRegionFromHash());
  });

  if (!toggle) {
    return;
  }

  toggle.addEventListener("click", function () {
    var currentTheme = root.getAttribute("data-theme");
    var nextTheme = currentTheme === "light" ? "dark" : "light";

    applyTheme(nextTheme);
    saveTheme(nextTheme);
  });
})();
