import colorsys

def generate_color_gradient_hsv(initial_color, final_color, num_steps , take_shorter_h_path = True):
    gradient = []
    initial_hsv = colorsys.rgb_to_hsv(*initial_color)
    final_hsv = colorsys.rgb_to_hsv(*final_color)

    h_diff = (final_hsv[0] - initial_hsv[0])
    if take_shorter_h_path:
        if h_diff > 0.5:
            h_diff = (initial_hsv[0] - final_hsv[0])
            h_diff %= 1
            h_diff *= -1

    for step in range(num_steps):
        t = step / (num_steps - 1)
        h = (initial_hsv[0] + t * (h_diff)) % 1
        s = initial_hsv[1] + t * (final_hsv[1] - initial_hsv[1])
        v = initial_hsv[2] + t * (final_hsv[2] - initial_hsv[2])
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        gradient.append([r, g, b, 1])
    return gradient

def generate_color_gradient(initial_color, final_color, num_steps):
    gradient = []
    for step in range(num_steps):
        t = step / (num_steps - 1)
        r = initial_color[0] + t * (final_color[0] - initial_color[0])
        g = initial_color[1] + t * (final_color[1] - initial_color[1])
        b = initial_color[2] + t * (final_color[2] - initial_color[2])
        gradient.append([r, g, b, 1])
    return gradient