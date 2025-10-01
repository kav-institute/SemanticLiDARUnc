import numpy as np

id_map = {
  0 : 0,     # "unlabeled"
  1 : 0,    # "outlier" mapped to "unlabeled" --------------------------mapped
  9: 0,
  10: 1,     # "car"
  11: 2,     # "bicycle"
  13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
  15: 3,     # "motorcycle"
  16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
  18: 4,     # "truck"
  20: 5,     # "other-vehicle"
  30: 6,     # "person"
  31: 7,     # "bicyclist"
  32: 8,     # "motorcyclist"
  40: 9,     # "road"
  44: 10,    # "parking"
  48: 11,    # "sidewalk"
  49: 12,    # "other-ground"
  50: 13,    # "building"
  51: 14,    # "fence"
  52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
  60: 19,     # "lane-marking" to "traffic-sign" ------------------------mapped
  70: 15,    # "vegetation"
  71: 16,    # "trunk"
  72: 17,    # "terrain"
  80: 18,    # "pole"
  81: 19,    # "traffic-sign"
  99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
  252: 1,    # "moving-car" to "car" ------------------------------------mapped
  253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
  254: 6,    # "moving-person" to "person" ------------------------------mapped
  255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
  256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
  257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
  258: 4,    # "moving-truck" to "truck" --------------------------------mapped
  259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

id_map_reduced = {
  0 : 0,     # "unlabeled"
  1 : 0,    # "outlier" mapped to "unlabeled" --------------------------mapped
  9: 0,
  10: 1,     # "car"
  11: 2,     # "bicycle" mapped to "two-wheeled -------------------mapped
  13: 3,     # "bus" mapped to "other-vehicle" --------------------------mapped
  15: 2,     # "motorcycle" mapped to "two-wheeled -------------------mapped
  16: 3,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
  18: 3,     # "truck" mapped to "other-vehicle" ---------------------mapped
  20: 3,     # "other-vehicle"
  30: 4,     # "person" 
  31: 5,     # "bicyclist" mapped to "rider" ---------------------mapped
  32: 5,     # "motorcyclist" mapped to "rider" ---------------------mapped
  40: 6,     # "road"
  44: 6,    # "parking"
  48: 7,    # "sidewalk"
  49: 8,    # "other-ground"
  50: 9,    # "building"
  51: 9,    # "fence"
  52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
  60: 6,     # "lane-marking" to "road" ---------------------------------mapped
  70: 7,    # "vegetation"
  71: 7,    # "trunk"
  72: 10,    # "terrain"
  80: 11,    # "pole"
  81: 12,    # "traffic-sign"
  99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
  252: 1,    # "moving-car" to "car" ------------------------------------mapped
  253: 5,    # "moving-bicyclist" to "rider" ------------------------mapped
  254: 6,    # "moving-person" to "person" ------------------------------mapped
  255: 5,    # "moving-motorcyclist" to "rider" ------------------mapped
  256: 3,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
  257: 3,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
  258: 3,    # "moving-truck" to "other-vehicle"" -----------------------mapped
  259: 3,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

id_map_dynamic = {
  0 : 0,     # "unlabeled"
  1 : 0,    # "outlier" mapped to "unlabeled" --------------------------mapped
  10: 1,     # "car"
  11: 2,     # "bicycle"
  13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
  15: 3,     # "motorcycle"
  16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
  18: 4,     # "truck"
  20: 5,     # "other-vehicle"
  30: 6,     # "person"
  31: 7,     # "bicyclist"
  32: 8,     # "motorcyclist"
  40: 0,     # "road"
  44: 0,    # "parking"
  48: 0,    # "sidewalk"
  49: 0,    # "other-ground"
  50: 0,    # "building"
  51: 0,    # "fence"
  52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
  60: 0,     # "lane-marking" to "road" ---------------------------------mapped
  70: 0,    # "vegetation"
  71: 0,    # "trunk"
  72: 0,    # "terrain"
  80: 0,    # "pole"
  81: 0,    # "traffic-sign"
  99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
  252: 1,    # "moving-car" to "car" ------------------------------------mapped
  253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
  254: 6,    # "moving-person" to "person" ------------------------------mapped
  255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
  256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
  257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
  258: 4,    # "moving-truck" to "truck" --------------------------------mapped
  259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

color_map = {
  0 : [0, 0, 0],
  1 : [245, 150, 100],
  2 : [245, 230, 100],
  3 : [150, 60, 30],
  4 : [180, 30, 80],
  5 : [255, 0, 0],
  6: [30, 30, 255],
  7: [200, 40, 255],
  8: [90, 30, 150],
  9: [125,125,125],#[255, 0, 255],
  10: [255, 150, 255],
  11: [75, 0, 75],
  12: [75, 0, 175],
  13: [0, 200, 255],
  14: [50, 120, 255],
  15: [0, 175, 0],
  16: [0, 60, 135],
  17: [80, 240, 150],
  18: [150, 240, 255],
  19: [250, 10, 250],
  20: [255, 255, 2] # TODO: potentially for SemanticSTF [255, 255, 255]
}

color_map_reduced = {
  0 : [0, 0, 0], # none
  1 : [245, 150, 100], # car
  2 : [245, 230, 100], # two-wheeled
  3 : [255, 0, 0], # other-vehicle
  4 : [30, 30, 255], # person
  5 : [200, 40, 255], # rider
  6: [125,125,125], # road
  7: [75, 0, 75], # sidewalk
  8: [255, 150, 255], # other-ground
  9: [0, 175, 0], # vegetation
  10: [0, 60, 135], # terrain
  11: [150, 240, 255], # pole
  12: [250, 250, 250] # traffic-sign
}

class_names = {
  0 : "unlabeled",
  1 : "car",
  2 : "bicycle",
  3 : "motorcycle",
  4 : "truck",
  5 : "other-vehicle",
  6: "person",
  7: "bicyclist",
  8: "motorcyclist",
  9: "road",
  10: "parking",
  11: "sidewalk",
  12: "other-ground",
  13: "building",
  14: "fence",
  15: "vegetation",
  16: "trunk",
  17: "terrain",
  18: "pole",
  19: "traffic-sign",
  20: "snow"
}

# color_map = {
#   0 : [0, 0, 0],
#   1 : [245, 150, 100],
#   2 : [245, 230, 100],
#   3 : [150, 60, 30],
#   4 : [180, 30, 80],
#   5 : [255, 0, 0],
#   6: [30, 30, 255],
#   7: [200, 40, 255],
#   8: [90, 30, 150],
#   9: [125,125,125],
#   10: [255, 150, 255],
#   11: [75, 0, 75],
#   12: [75, 0, 175],
#   13: [0, 200, 255],
#   14: [50, 120, 255],
#   15: [0, 175, 0],
#   16: [0, 60, 135],
#   17: [80, 240, 150],
#   18: [150, 240, 255],
#   19: [250, 250, 250],
#   20: [0, 250, 0]
# }

# Create the custom color map
custom_colormap = np.zeros((256, 1, 3), dtype=np.uint8)

for i in range(256):
    if i in color_map:
        custom_colormap[i, 0, :] = color_map[i]
    else:
        # If the index is not defined in the color map, set it to black
        custom_colormap[i, 0, :] = [0, 0, 0]
custom_colormap = custom_colormap[...,::-1]
