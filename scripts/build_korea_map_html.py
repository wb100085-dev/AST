# -*- coding: utf-8 -*-
"""실제 한국 행정구역 SVG path로 assets/korea_sido_map.html 생성"""
import urllib.request
import re
import json
import os

ID_TO_CODE = {
    'seoul': '11', 'busan': '21', 'daegu': '22', 'incheon': '23', 'gwangju': '24',
    'daejeon': '25', 'ulsan': '26', 'sejong': '29', 'gyeonggi': '31', 'gangwon': '32',
    'north-chungcheong': '33', 'south-chungcheong': '34', 'north-jeolla': '35',
    'south-jeolla': '36', 'north-gyeongsang': '37', 'south-gyeongsang': '38', 'jeju': '39'
}
CODE_TO_NAME = {
    '11': '서울특별시', '21': '부산광역시', '22': '대구광역시', '23': '인천광역시', '24': '광주광역시',
    '25': '대전광역시', '26': '울산광역시', '29': '세종특별자치시', '31': '경기도', '32': '강원도',
    '33': '충청북도', '34': '충청남도', '35': '전라북도', '36': '전라남도', '37': '경상북도',
    '38': '경상남도', '39': '제주특별자치도'
}

def main():
    url = 'https://raw.githubusercontent.com/VictorCazanave/svg-maps/master/packages/south-korea/index.js'
    with urllib.request.urlopen(url, timeout=15) as r:
        raw = r.read().decode('utf-8')
    raw = raw.strip()
    if raw.startswith('export default '):
        raw = raw[14:]
    if raw.endswith(';'):
        raw = raw[:-1]
    data = json.loads(raw)
    view_box = data.get('viewBox', '0 0 524 631')
    locations = data.get('locations', [])

    sido_paths = []
    for loc in locations:
        lid = loc.get('id', '')
        code = ID_TO_CODE.get(lid)
        if not code:
            continue
        name = CODE_TO_NAME.get(code, loc.get('name', ''))
        d = loc.get('path', '')
        if d:
            sido_paths.append({'code': code, 'name': name, 'd': d})

    paths_json = json.dumps(sido_paths, ensure_ascii=False)

    html = '''<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; padding: 8px; background: transparent; font-family: inherit; }
    .map-title { font-size: 0.875rem; font-weight: 600; color: #334155; margin-bottom: 8px; }
    .svg-wrap { width: 100%; max-height: 320px; border: 1px solid #e2e8f0; border-radius: 8px; background: #fff; }
    .svg-wrap svg { width: 100%; height: auto; display: block; }
    .sido-path { cursor: pointer; transition: filter 0.2s, stroke-width 0.2s; }
    .sido-path:hover { filter: brightness(1.1); }
    .sido-path.selected { filter: brightness(1.15); stroke-width: 2; }
  </style>
</head>
<body>
  <div id="root"></div>
  <script>
(function() {
  var SIDO_PATHS = ''' + paths_json + ''';
  var VIEW_BOX = "''' + view_box + '''";
  var selectedCode = (typeof window.__SIDO_CODE__ !== "undefined" ? window.__SIDO_CODE__ : "") || "";
  var regionStats = (typeof window.__REGION_STATS__ !== "undefined" ? window.__REGION_STATS__ : {}) || {};
  var hoverCode = null;
  function getFill(code) {
    if (selectedCode === code) return "#4f46e5";
    if (hoverCode === code) return "#818cf8";
    return "#c7d2fe";
  }
  function getStrokeWidth(code) {
    return selectedCode === code ? 2 : 0.8;
  }
  function escapeHtml(s) {
    var div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }
  function navigateToSido(code) {
    try {
      var base = window.parent.location.pathname || "/";
      var sep = base.indexOf("?") >= 0 ? "&" : "?";
      window.parent.location = base + sep + "sido_code=" + encodeURIComponent(code);
    } catch (e) {
      selectedCode = code;
      update();
    }
  }
  function update() {
    var mapPaths = "";
    for (var i = 0; i < SIDO_PATHS.length; i++) {
      var p = SIDO_PATHS[i];
      mapPaths += '<path class="sido-path' + (selectedCode === p.code ? ' selected' : '') + '" d="' + p.d + '" fill="' + getFill(p.code) + '" stroke="#3730a3" stroke-width="' + getStrokeWidth(p.code) + '" data-sido-code="' + escapeHtml(p.code) + '" title="' + escapeHtml(p.name) + '"/>';
    }
    var svg = '<svg viewBox="' + VIEW_BOX + '" xmlns="http://www.w3.org/2000/svg">' + mapPaths + '</svg>';
    root.innerHTML = '<div class="map-title">대한민국 행정구역 (도 단위)</div><div class="svg-wrap">' + svg + '</div>';
    var paths = root.querySelectorAll(".sido-path");
    for (var j = 0; j < paths.length; j++) {
      (function(code) {
        paths[j].addEventListener("click", function() { navigateToSido(code); });
        paths[j].addEventListener("mouseenter", function() { hoverCode = code; update(); });
        paths[j].addEventListener("mouseleave", function() { hoverCode = null; update(); });
      })(SIDO_PATHS[j].code);
    }
  }
  var root = document.getElementById("root");
  if (root) update();
})();
  </script>
</body>
</html>
'''

    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')
    out_path = os.path.join(assets_dir, 'korea_sido_map.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print('Written', out_path)


if __name__ == '__main__':
    main()
