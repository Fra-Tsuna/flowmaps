# lookup tables for the 2d dataset

FURNITURE_TYPES = {
        "kitchen_table": (0, 0, 0),           # black
        "office_desk": (205, 133, 63),       # lighter brown
        "dining_table": (80, 50, 85)          # lighter purple
    }

OBJECT_TYPES = {
    "fork":   {"shape": "square",   "color": (255, 0, 0),   "furniture": "kitchen_table"},
    "bottle": {"shape": "circle",   "color": (0, 255, 0),   "furniture": "dining_table"},
    "remote": {"shape": "triangle", "color": (0, 0, 255),   "furniture": "office_desk"}
}

COLOR2ID = {
    (255, 0, 0): 0,
    (0, 255, 0): 1,
    (0, 0, 255): 2,
    (0, 0, 0): 3,
    (205, 133, 63): 4,
    (80, 50, 85): 5,
}

ID2COLOR = {v: k for k, v in COLOR2ID.items()}