import colorsys

class getColors:
    def __init__(self, num_colors, bg_color, axis_color, class_names, default_colors=None, color_names=None, benign_malignant=False):
        self.bg_color = [x / 255.0 for x in bg_color]  # Normalize to [0, 1]
        self.axis_color = [x / 255.0 for x in axis_color]  # Normalize to [0, 1]
        self.colors_array = []
        self.colors_names_array = []
        if default_colors is not None:
            self.colors_array = default_colors
            if color_names is not None:
                self.colors_names_array = color_names
        self.num_colors = num_colors
        
        if benign_malignant:
            # for each class in class_names, assign a color with generate colors or use red for malignant and green for benign
            for i in range(len(class_names)):
                name = class_names[i].lower()
                if 'benign' in name or 'positive' in name:
                    self.colors_array.append([0, 255, 0])
                    self.colors_names_array.append('Green')
                    self.num_colors -= 1
                elif 'malignant' in name or 'negative' in name:
                    self.colors_array.append([255, 0, 0])
                    self.colors_names_array.append('Red')
                    self.num_colors -= 1
        if benign_malignant and self.num_colors > 0:
            # Generate the remaining colors using the red / green colors as defaults
            self.__init__(self.num_colors, bg_color, axis_color, class_names, self.colors_array, self.colors_names_array, benign_malignant=False)
        else:
            self.generate_colors()
    def generate_colors(self):
        for i in range(self.num_colors):
            hue = i / float(self.num_colors)
            lightness = 0.5
            saturation = 0.8

            r, g, b = [int(x * 255.0) for x in colorsys.hls_to_rgb(hue, lightness, saturation)]

            self.colors_array.append([r, g, b])
            self.colors_names_array.append(f"color_{i}")

def shift_hue(rgb, amount):
    # Convert RGB to HSV
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    # Shift the hue
    h = (h + amount) % 1.0
    # Convert back to RGB
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)
