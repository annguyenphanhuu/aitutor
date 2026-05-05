"""Grade 12 Math skill graph with prerequisite dependencies."""

# ── Skill Definitions ────────────────────────────────────────────────────────
# 6 chapters matching actual THPT exam scope (GDPT 2018):
#   1. Đạo hàm
#   2. Nguyên hàm và Tích phân
#   3. Dãy số – Cấp số cộng – Cấp số nhân
#   4. Hình học không gian
#   5. Tổ hợp – Xác suất
#   6. Thống kê

SKILLS = {
    # ── 1. Đạo hàm ──────────────────────────────────────────────────────────
    "derivative_basic": {
        "name": "Đạo hàm cơ bản",
        "chapter": "Đạo hàm",
        "description": "Tính đạo hàm các hàm số cơ bản: đa thức, lượng giác, mũ, log",
        "prerequisites": [],
    },
    "derivative_rules": {
        "name": "Quy tắc đạo hàm",
        "chapter": "Đạo hàm",
        "description": "Đạo hàm tích, thương, hàm hợp",
        "prerequisites": ["derivative_basic"],
    },
    "derivative_applications": {
        "name": "Ứng dụng đạo hàm",
        "chapter": "Đạo hàm",
        "description": "Cực trị, GTLN–GTNN, sự đơn điệu, tiếp tuyến",
        "prerequisites": ["derivative_rules"],
    },
    "function_survey": {
        "name": "Khảo sát hàm số",
        "chapter": "Đạo hàm",
        "description": "Bảng biến thiên, đồ thị hàm số bậc 3, bậc 4, trùng phương, phân thức",
        "prerequisites": ["derivative_applications"],
    },

    # ── 2. Nguyên hàm và Tích phân ──────────────────────────────────────────
    "primitive_basic": {
        "name": "Nguyên hàm cơ bản",
        "chapter": "Nguyên hàm và Tích phân",
        "description": "Nguyên hàm, bảng nguyên hàm, tích phân bất định",
        "prerequisites": ["derivative_basic"],
    },
    "integral_definite": {
        "name": "Tích phân xác định",
        "chapter": "Nguyên hàm và Tích phân",
        "description": "Tính tích phân xác định bằng Newton–Leibniz, đổi biến, từng phần",
        "prerequisites": ["primitive_basic"],
    },
    "integral_applications": {
        "name": "Ứng dụng tích phân",
        "chapter": "Nguyên hàm và Tích phân",
        "description": "Diện tích hình phẳng, thể tích vật tròn xoay",
        "prerequisites": ["integral_definite"],
    },

    # ── 3. Dãy số – Cấp số cộng – Cấp số nhân ───────────────────────────────
    "sequence_basic": {
        "name": "Dãy số cơ bản",
        "chapter": "Dãy số",
        "description": "Khái niệm dãy số, số hạng tổng quát, tính đơn điệu, dãy bị chặn",
        "prerequisites": [],
    },
    "arithmetic_sequence": {
        "name": "Cấp số cộng",
        "chapter": "Dãy số",
        "description": "Định nghĩa CSC, số hạng thứ n, tổng n số hạng đầu, chèn CSC",
        "prerequisites": ["sequence_basic"],
    },
    "geometric_sequence": {
        "name": "Cấp số nhân",
        "chapter": "Dãy số",
        "description": "Định nghĩa CSN, số hạng thứ n, tổng n số hạng đầu, CSN vô hạn",
        "prerequisites": ["sequence_basic"],
    },

    # ── 4. Hình học không gian ───────────────────────────────────────────────
    "geometry_vectors": {
        "name": "Vectơ trong không gian",
        "chapter": "Hình học không gian",
        "description": "Tọa độ vectơ, tích vô hướng, tích có hướng trong Oxyz",
        "prerequisites": [],
    },
    "geometry_line_plane": {
        "name": "Đường thẳng và mặt phẳng",
        "chapter": "Hình học không gian",
        "description": "Phương trình tham số đường thẳng, phương trình tổng quát mặt phẳng",
        "prerequisites": ["geometry_vectors"],
    },
    "geometry_distance_angle": {
        "name": "Khoảng cách và góc",
        "chapter": "Hình học không gian",
        "description": "Khoảng cách điểm–mp, đường thẳng–mp; góc giữa đường thẳng và mp",
        "prerequisites": ["geometry_line_plane"],
    },
    "geometry_sphere": {
        "name": "Mặt cầu",
        "chapter": "Hình học không gian",
        "description": "Phương trình mặt cầu, tâm bán kính, tiếp tuyến, thiết diện",
        "prerequisites": ["geometry_line_plane"],
    },

    # ── 5. Tổ hợp – Xác suất ────────────────────────────────────────────────
    "combinatorics_basic": {
        "name": "Tổ hợp – Chỉnh hợp – Hoán vị",
        "chapter": "Tổ hợp – Xác suất",
        "description": "Quy tắc đếm, hoán vị, chỉnh hợp, tổ hợp, nhị thức Newton",
        "prerequisites": [],
    },
    "probability_basic": {
        "name": "Xác suất cơ bản",
        "chapter": "Tổ hợp – Xác suất",
        "description": "Biến cố, xác suất cổ điển, quy tắc cộng nhân, xác suất có điều kiện",
        "prerequisites": ["combinatorics_basic"],
    },
    "probability_distribution": {
        "name": "Phân phối xác suất",
        "chapter": "Tổ hợp – Xác suất",
        "description": "Biến ngẫu nhiên rời rạc, phân phối nhị thức, kỳ vọng, phương sai",
        "prerequisites": ["probability_basic"],
    },

    # ── 6. Thống kê ─────────────────────────────────────────────────────────
    "statistics_descriptive": {
        "name": "Thống kê mô tả",
        "chapter": "Thống kê",
        "description": "Bảng số liệu, biểu đồ, trung bình, trung vị, mốt, tứ phân vị",
        "prerequisites": [],
    },
    "statistics_inference": {
        "name": "Thống kê suy luận",
        "chapter": "Thống kê",
        "description": "Mẫu số liệu, tần số, tần suất, độ lệch chuẩn, ước lượng tham số",
        "prerequisites": ["statistics_descriptive"],
    },
}


def get_skill(skill_id: str) -> dict | None:
    """Get skill definition by ID."""
    return SKILLS.get(skill_id)


def get_all_skills() -> dict:
    """Get all skill definitions."""
    return SKILLS


def get_prerequisites(skill_id: str) -> list[str]:
    """Get prerequisite skill IDs for a given skill."""
    skill = SKILLS.get(skill_id)
    if not skill:
        return []
    return list(skill["prerequisites"])


def get_deep_prerequisites(skill_id: str) -> list[str]:
    """Get all transitive prerequisites (BFS)."""
    visited = set()
    queue   = list(get_prerequisites(skill_id))
    result  = []

    while queue:
        current = queue.pop(0)
        if current not in visited:
            visited.add(current)
            result.append(current)
            queue.extend(get_prerequisites(current))

    return result


def get_skills_by_chapter(chapter: str) -> list[str]:
    """Get all skill_ids belonging to a specific chapter."""
    return [sid for sid, info in SKILLS.items() if info["chapter"] == chapter]


def get_chapters() -> list[str]:
    """Get all unique chapter names (ordered)."""
    seen: list[str] = []
    for s in SKILLS.values():
        c = str(s["chapter"])
        if c not in seen:
            seen.append(c)
    return seen


def find_weak_prerequisites(
    masteries: dict[str, float], skill_id: str, threshold: float = 0.5
) -> list[str]:
    """
    Given current mastery levels and a target skill,
    find prerequisite skills that are below the mastery threshold.
    """
    prereqs = get_deep_prerequisites(skill_id)
    return [p for p in prereqs if masteries.get(p, 0.0) < threshold]
