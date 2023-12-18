import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
import numpy as np
import cv2
import math

st.set_page_config(page_title="Color Blindness Web Application", layout="wide")

# Color names and corresponding RGB values
colors_data = {
    "Red": {"min": (200, 0, 0), "max": (255, 50, 50)},
    "Yellow": {"min": (200, 200, 0), "max": (255, 255, 50)},
    "Blue": {"min": (0, 0, 200), "max": (50, 50, 255)},
    "Green": {"min": (0, 200, 0), "max": (50, 255, 50)},
    "Orange": {"min": (200, 100, 0), "max": (255, 200, 50)},
    "Purple": {"min": (80, 0, 80), "max": (150, 50, 150)},
    "Black": {"min": (0, 0, 0), "max": (50, 50, 50)},
    "White": {"min": (200, 200, 200), "max": (255, 255, 255)},
    "Gray": {"min": (100, 100, 100), "max": (180, 180, 180)},
    "Brown": {"min": (100, 40, 40), "max": (180, 80, 80)},
    "Pink": {"min": (200, 150, 150), "max": (255, 200, 200)},
    "Violet": {"min": (180, 100, 180), "max": (255, 150, 255)}
}

def get_average_pixel_color(image, x, y, size=10):
    half_size = size // 2
    region = image.crop((x - half_size, y - half_size, x + half_size, y + half_size))

    # Check if region.getdata() is an integer
    if isinstance(region.getpixel((0, 0)), int):
        # If it's an integer, return a tuple with the integer value for each channel
        return (region.getpixel((0, 0)), region.getpixel((0, 0)), region.getpixel((0, 0)))

    # If it's not an integer, calculate the average color
    average_color = tuple(int(sum(channel) / size**2) for channel in zip(*region.getdata()))
    return average_color


def cie76_color_difference(color1, color2):
    # Calculate the CIE76 color difference between two RGB colors
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(color1, color2)))

def get_closest_color_name(rgb_value):
    closest_color_name = None
    min_distance = float("inf")

    for color_name, color_range in colors_data.items():
        min_values = color_range["min"]
        max_values = color_range["max"]

        # Check if the RGB value is within the defined range
        if all(min_v <= value <= max_v for value, min_v, max_v in zip(rgb_value, min_values, max_values)):
            closest_color_name = color_name
            break

        # Calculate the CIE76 color difference to the color range
        distance = cie76_color_difference(rgb_value, [(min_v + max_v) / 2 for min_v, max_v in zip(min_values, max_values)])

        if distance < min_distance:
            min_distance = distance
            closest_color_name = color_name

    return closest_color_name
def rgb_to_lms(img):
    # Make sure the input image array has the expected shape
    
    # RGB to LMS matrix
    lms_matrix = np.array([[0.4002, 0.7075, -0.0808],
                           [-0.2263, 1.1653, 0.0457],
                           [0.0000, 0.0000, 0.9182]])

    lms_matrix = np.array(
        [[0.3904725 , 0.54990437, 0.00890159],
        [0.07092586, 0.96310739, 0.00135809],
        [0.02314268, 0.12801221, 0.93605194]]
        )
    return np.tensordot(img, lms_matrix, axes=([2], [1]))
    
def lms_to_rgb(img):
    """
    rgb_matrix = np.array(
        [[0.0809444479, -0.130504409, 0.116721066],
        [0.113614708, -0.0102485335, 0.0540193266],
        [-0.000365296938, -0.00412161469, 0.693511405]
        ]
        )
    """
    rgb_matrix = np.array(
        [[ 2.85831110e+00, -1.62870796e+00, -2.48186967e-02],
        [-2.10434776e-01,  1.15841493e+00,  3.20463334e-04],
        [-4.18895045e-02, -1.18154333e-01,  1.06888657e+00]]
        )
    return np.tensordot(img, rgb_matrix, axes=([2], [1]))

def simulate_colorblindness(img, colorblind_type):
    lms_img = rgb_to_lms(img)
    if colorblind_type.lower() in ['protanopia', 'p', 'pro']:
        sim_matrix = np.array([[0, 0.90822864, 0.008192], [0, 1, 0], [0, 0, 1]], dtype=np.float16)
    elif colorblind_type.lower() in ['deuteranopia', 'd', 'deut']:
        sim_matrix =  np.array([[1, 0, 0], [1.10104433,  0, -0.00901975], [0, 0, 1]], dtype=np.float16)
    elif colorblind_type.lower() in ['tritanopia', 't', 'tri']:
        sim_matrix = np.array([[1, 0, 0], [0, 1, 0], [-0.15773032,  1.19465634, 0]], dtype=np.float16)
    else:
        raise ValueError('{} is an unrecognized colorblindness type.'.format(colorblind_type))
    lms_img = np.tensordot(lms_img, sim_matrix, axes=([2], [1]))
    rgb_img = lms_to_rgb(lms_img)
    return rgb_img.astype(np.uint8)


def main():
    st.title("Color Recognition App")
    st.divider()
    colupload, coloption = st.columns(2)
    
    with colupload:
        st.subheader("Step 1: Select an image",divider="gray")
        uploaded_file = st.file_uploader("Upload an image (Suggested Dimensions: 800x750)", type=["jpg", "jpeg", "png"])

    
    with coloption:
        st.subheader("Step 2: Choose a type of color blindness",divider="gray")
        option = st.selectbox(
                    "Select type of Color Blindness",
                    ("Protanopia", "Tritanopia", "Deuteranopia", "Monochromacy"),
                    index=None,
                    placeholder="Select contact method...",
                )
    st.subheader("Results",divider="rainbow")
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Step 3: Click a pixel in the image", divider="gray")
            value = streamlit_image_coordinates(image)

        # Initialize variables outside the conditional blocks
        average_pixel_color = None
        average_pixel_color_blind = None
        closest_color_name = None
        closest_color_name_blind = None
        simulated_img = None

        if value is not None:
            x = int(value['x'])
            y = int(value['y'])
            average_pixel_color = get_average_pixel_color(image, x, y, size=10)
            closest_color_name = get_closest_color_name(average_pixel_color)

            # Inside the main function, after applying colorblind simulation
            if option == "Protanopia":
                simulated_img = simulate_colorblindness(image, colorblind_type='protanopia')
            elif option == "Tritanopia":
                simulated_img = simulate_colorblindness(image, colorblind_type='tritanopia')
            elif option == "Deuteranopia":
                simulated_img = simulate_colorblindness(image, colorblind_type='deuteranopia')
            elif option == "Monochromacy":
                # Convert PIL Image to NumPy array
                image_array = np.array(image)
                # Convert to grayscale
                simulated_img = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

            # Check if simulated_img is not None before converting it to a PIL Image
            if simulated_img is not None:
                with col2:
                    st.subheader("Step 4: Observe the marker", divider="gray")
                    # Convert the NumPy array to a PIL Image
                    simulated_imgs = Image.fromarray(simulated_img)

                    average_pixel_color_blind = get_average_pixel_color(simulated_imgs, x, y, size=15)
                    closest_color_name_blind = get_closest_color_name(average_pixel_color_blind)

                    # Display the selected region with the rectangle drawn around it in the right column
                    draw = ImageDraw.Draw(simulated_imgs)
                    draw.rectangle([x - 15, y - 15, x + 15, y + 15], outline="red", width=5)
                    col2.image(simulated_imgs, caption="Selected Region", use_column_width=True)

        
            st.subheader("Step 5: Check the RBG color values", divider="gray")
            col3, col4 = st.columns(2)
            with col3:

                if average_pixel_color is not None:
                
                    wch_colour_box = (average_pixel_color[0], average_pixel_color[1], average_pixel_color[2])
                    wch_colour_font = (238,238,228)
                    fontsize = 25
                    valign = "left"
                    iconname = "fas fa-asterisk"
                    
                    htmlstr = f"""<p style='background-color: rgb({wch_colour_box[0]}, 
                                                                {wch_colour_box[1]}, 
                                                                {wch_colour_box[2]}, 0.75); 
                                            color: rgb({wch_colour_font[0]}, 
                                                    {wch_colour_font[1]}, 
                                                    {wch_colour_font[2]}, 0.75); 
                                            font-size: {fontsize}px; 
                                            border-radius: 7px; 
                                            padding-left: 12px; 
                                            padding-top: 18px; 
                                            padding-bottom: 18px; 
                                            line-height:65px;'>
                                            <i class='{iconname} fa-xs'></i><b> {closest_color_name} </b> 
                                            </style>
                                            <BR>
                                            <span style='font-size: 16px;'>{f"Average RGB: {average_pixel_color[0]}, {average_pixel_color[1]}, {average_pixel_color[2]}"}</style></span></p>"""

                    st.markdown(htmlstr, unsafe_allow_html=True)
            
            with col4:
                if average_pixel_color_blind is not None:
                    wch_colour_box = (average_pixel_color_blind[0], average_pixel_color_blind[1], average_pixel_color_blind[2])
                    wch_colour_font = (238,238,228)
                    fontsize = 25
                    valign = "left"
                    iconname = "fas fa-asterisk"
                    
                    htmlstr = f"""<p style='background-color: rgb({wch_colour_box[0]}, 
                                                                {wch_colour_box[1]}, 
                                                                {wch_colour_box[2]}, 0.75); 
                                            color: rgb({wch_colour_font[0]}, 
                                                    {wch_colour_font[1]}, 
                                                    {wch_colour_font[2]}, 0.75); 
                                            font-size: {fontsize}px; 
                                            border-radius: 7px; 
                                            padding-left: 12px; 
                                            padding-top: 18px; 
                                            padding-bottom: 18px; 
                                            line-height:65px;'>
                                            <i class='{iconname} fa-xs'></i><b> {closest_color_name_blind} </b> 
                                            </style>
                                            <BR>
                                            <span style='font-size: 16px;'>{f"Average RGB: {average_pixel_color_blind[0]}, {average_pixel_color_blind[1]}, {average_pixel_color_blind[2]}"}</style></span></p>"""

                    st.markdown(htmlstr, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
