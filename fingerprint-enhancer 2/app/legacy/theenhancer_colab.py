import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
from google.colab import files
import os
from scipy import ndimage
from skimage import filters, morphology, exposure
import warnings
warnings.filterwarnings('ignore')

print("ENHANCED FINGERPRINT TOOL WITH SELECTION & BMP EXPORT")
print("=" * 70)
print("Fixed blank output issue!")
print("Now produces visible enhanced fingerprints")
print("Added numbering for multiple results")
print("Save selected result as BMP file")
print("=" * 70)

class EnhancedFingerprintSelector:
    def __init__(self):
        self.methods = {
            'enhanced_gabor': 'Enhanced Gabor Filter (Fixed)',
            'clahe_adaptive': 'CLAHE + Adaptive Threshold',
            'ridge_enhancement': 'Ridge Pattern Enhancement',
            'contrast_stretch': 'Contrast Stretching',
            'unsharp_mask': 'Unsharp Masking',
            'multi_scale': 'Multi-Scale Enhancement'
        }
        self.current_results = {}
        self.current_filename = ""

    def preprocess_image(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def enhanced_gabor_filter(self, image, show_steps=False):
        print("Applying FIXED Gabor Filter Enhancement...")

        image = self.preprocess_image(image)
        original = image.copy()

        if show_steps:
            plt.figure(figsize=(20, 5))
            step_count = 1

        normalized = cv2.normalize(image, None, 30, 220, cv2.NORM_MINMAX).astype(np.uint8)

        if show_steps:
            plt.subplot(1, 5, step_count)
            plt.imshow(normalized, cmap='gray')
            plt.title('1. Normalized')
            plt.axis('off')
            step_count += 1

        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
        clahe_applied = clahe.apply(normalized)

        if show_steps:
            plt.subplot(1, 5, step_count)
            plt.imshow(clahe_applied, cmap='gray')
            plt.title('2. CLAHE Enhanced')
            plt.axis('off')
            step_count += 1

        orientations = [0, 45, 90, 135]
        gabor_responses = []

        for angle in orientations:
            theta = np.deg2rad(angle)
            kernel = cv2.getGaborKernel(
                (15, 15), 3, theta, 8, 0.8, 0, ktype=cv2.CV_32F
            )
            filtered = cv2.filter2D(clahe_applied, cv2.CV_32F, kernel)
            filtered = np.abs(filtered)
            filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
            gabor_responses.append(filtered)

        gabor_combined = np.mean(gabor_responses, axis=0).astype(np.uint8)

        if show_steps:
            plt.subplot(1, 5, step_count)
            plt.imshow(gabor_combined, cmap='gray')
            plt.title('3. Gabor Filtered')
            plt.axis('off')
            step_count += 1

        contrast_enhanced = cv2.convertScaleAbs(gabor_combined, alpha=1.2, beta=10)

        if show_steps:
            plt.subplot(1, 5, step_count)
            plt.imshow(contrast_enhanced, cmap='gray')
            plt.title('4. Contrast Enhanced')
            plt.axis('off')
            step_count += 1

        _, binary = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

        if show_steps:
            plt.subplot(1, 5, step_count)
            plt.imshow(cleaned, cmap='gray')
            plt.title('5. Final Result')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        print("Fixed Gabor enhancement completed!")
        return cleaned

    def clahe_adaptive_enhancement(self, image):
        print("Applying CLAHE + Adaptive Enhancement...")
        image = self.preprocess_image(image)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_applied = clahe.apply(image)
        blurred = cv2.GaussianBlur(clahe_applied, (3, 3), 0)
        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        print("CLAHE + Adaptive enhancement completed!")
        return adaptive

    def ridge_pattern_enhancement(self, image):
        print("Applying Ridge Pattern Enhancement...")
        image = self.preprocess_image(image)
        
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        enhanced = cv2.addWeighted(image, 0.7, magnitude, 0.3, 0)
        _, thresholded = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print("Ridge pattern enhancement completed!")
        return thresholded

    def contrast_stretching(self, image):
        print("Applying Contrast Stretching...")
        image = self.preprocess_image(image)
        
        p2, p98 = np.percentile(image, (2, 98))
        stretched = exposure.rescale_intensity(image, in_range=(p2, p98))
        stretched = (stretched * 255).astype(np.uint8)
        _, binary = cv2.threshold(stretched, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print("Contrast stretching completed!")
        return binary

    def unsharp_masking(self, image):
        print("Applying Unsharp Masking...")
        image = self.preprocess_image(image)
        
        blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
        unsharp = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)
        _, thresholded = cv2.threshold(unsharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print("Unsharp masking completed!")
        return thresholded

    def multi_scale_enhancement(self, image):
        print("Applying Multi-Scale Enhancement...")
        image = self.preprocess_image(image)
        
        scales = [0.5, 1.0, 1.5]
        enhanced_scales = []

        for scale in scales:
            if scale != 1.0:
                h, w = image.shape
                new_h, new_w = int(h * scale), int(w * scale)
                scaled = cv2.resize(image, (new_w, new_h))
            else:
                scaled = image.copy()

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
            enhanced = clahe.apply(scaled)

            if scale != 1.0:
                enhanced = cv2.resize(enhanced, (w, h))

            enhanced_scales.append(enhanced)

        combined = np.mean(enhanced_scales, axis=0).astype(np.uint8)
        _, thresholded = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print("Multi-scale enhancement completed!")
        return thresholded

    def enhance_image(self, image, method='enhanced_gabor', show_steps=False):
        if method == 'enhanced_gabor':
            return self.enhanced_gabor_filter(image, show_steps)
        elif method == 'clahe_adaptive':
            return self.clahe_adaptive_enhancement(image)
        elif method == 'ridge_enhancement':
            return self.ridge_pattern_enhancement(image)
        elif method == 'contrast_stretch':
            return self.contrast_stretching(image)
        elif method == 'unsharp_mask':
            return self.unsharp_masking(image)
        elif method == 'multi_scale':
            return self.multi_scale_enhancement(image)
        else:
            print(f"Unknown method: {method}")
            return image

    def save_as_bmp(self, image, filename, method_name):
        base_name = os.path.splitext(filename)[0]
        bmp_filename = f"{base_name}_{method_name}_enhanced.bmp"
        
        cv2.imwrite(bmp_filename, image)
        
        print(f"Saved as: {bmp_filename}")
        return bmp_filename

    def display_numbered_results(self, original, results, filename):
        print(f"\nNUMBERED ENHANCEMENT RESULTS FOR: {filename}")
        print("=" * 70)
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Numbered Fingerprint Enhancement Results: {filename}', fontsize=16, fontweight='bold')

        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('0. Original Image', fontsize=12, fontweight='bold', color='blue')
        axes[0, 0].axis('off')

        positions = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]
        method_names = [
            '1. Fixed Gabor (Best)',
            '2. CLAHE + Adaptive',
            '3. Ridge Enhancement',
            '4. Contrast Stretching',
            '5. Unsharp Masking',
            '6. Multi-Scale'
        ]
        
        method_keys = list(self.methods.keys())

        for i, (method, name) in enumerate(zip(method_keys, method_names)):
            if i < len(positions):
                row, col = positions[i]
                axes[row, col].imshow(results[method], cmap='gray')
                axes[row, col].set_title(name, fontsize=10, fontweight='bold', color='red')
                axes[row, col].axis('off')

        axes[1, 3].axis('off')
        axes[1, 3].text(0.5, 0.5, 'Choose a number\n0-6 to select!', 
                       transform=axes[1, 3].transAxes, fontsize=14, 
                       ha='center', va='center', fontweight='bold')

        plt.tight_layout()
        plt.show()

        print("\nSELECTION MENU:")
        print("=" * 40)
        print("0  Original Image")
        print("1  Fixed Gabor Enhancement")
        print("2  CLAHE + Adaptive Threshold")
        print("3  Ridge Pattern Enhancement")
        print("4  Contrast Stretching")
        print("5  Unsharp Masking")
        print("6  Multi-Scale Enhancement")
        print("=" * 40)

def enhanced_upload_and_select():
    
    print("\nUPLOAD YOUR FINGERPRINT - ENHANCED WITH SELECTION!")
    print("=" * 70)
    print("1. Upload your fingerprint image")
    print("2. View all numbered enhancement results")
    print("3. Select your preferred result by number")
    print("4. Download as BMP file")
    print("=" * 70)

    uploaded = files.upload()

    if not uploaded:
        print("No files uploaded!")
        return

    enhancer = EnhancedFingerprintSelector()

    for filename in uploaded.keys():
        print(f"\nProcessing: {filename}")
        print("-" * 50)

        try:
            image = cv2.imread(filename)
            if image is None:
                image = np.array(Image.open(filename))
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            original = image.copy()
            if len(original.shape) == 3:
                original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            else:
                original_gray = original

            print(f"Image size: {image.shape[1]}x{image.shape[0]} pixels")

            results = {}
            methods_to_test = list(enhancer.methods.keys())

            print("\nApplying all enhancement methods...")
            for method in methods_to_test:
                enhanced = enhancer.enhance_image(image, method)
                results[method] = enhanced

            enhancer.current_results = results.copy()
            enhancer.current_results['original'] = original_gray
            enhancer.current_filename = filename

            enhancer.display_numbered_results(original_gray, results, filename)

            print(f"\nMAKE YOUR SELECTION:")
            print("Type: select_and_save(NUMBER)")
            print("Example: select_and_save(1)  # for Fixed Gabor")
            print("Available numbers: 0, 1, 2, 3, 4, 5, 6")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

def select_and_save(number):
    global _current_enhancer
    
    if not hasattr(_current_enhancer, 'current_results') or not _current_enhancer.current_results:
        print("No enhancement results available! Please run enhanced_upload_and_select() first.")
        return

    if number not in [0, 1, 2, 3, 4, 5, 6]:
        print("Invalid selection! Please choose a number from 0 to 6.")
        return

    selection_map = {
        0: ('original', 'Original'),
        1: ('enhanced_gabor', 'Fixed_Gabor'),
        2: ('clahe_adaptive', 'CLAHE_Adaptive'),
        3: ('ridge_enhancement', 'Ridge_Enhancement'),
        4: ('contrast_stretch', 'Contrast_Stretch'),
        5: ('unsharp_mask', 'Unsharp_Mask'),
        6: ('multi_scale', 'Multi_Scale')
    }

    method_key, method_name = selection_map[number]

    print(f"\nYou selected: {number} {method_name}")
    print("-" * 50)

    try:
        if method_key == 'original':
            selected_image = _current_enhancer.current_results['original']
        else:
            selected_image = _current_enhancer.current_results[method_key]

        plt.figure(figsize=(8, 8))
        plt.imshow(selected_image, cmap='gray')
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show()

        base_name = os.path.splitext(_current_enhancer.current_filename)[0]
        bmp_filename = f"{base_name}_{method_name}_enhanced.bmp"
        
        if selected_image.dtype != np.uint8:
            selected_image = selected_image.astype(np.uint8)
        
        success = cv2.imwrite(bmp_filename, selected_image)
        
        if success:
            print(f"BMP file saved successfully: {bmp_filename}")
            
            try:
                files.download(bmp_filename)
                print(f"Download started for: {bmp_filename}")
            except Exception as download_error:
                print(f"File saved but download failed: {download_error}")
                print(f"You can manually download '{bmp_filename}' from the file panel")
        else:
            print(f"Failed to save BMP file")
            return
            
        print(f"\nSUCCESS!")
        print(f"File: {bmp_filename}")
        print(f"Method: {method_name}")
        print(f"Size: {selected_image.shape}")

    except Exception as e:
        print(f"Error saving selection: {str(e)}")

def save_all_results():
    global _current_enhancer
    
    if not hasattr(_current_enhancer, 'current_results') or not _current_enhancer.current_results:
        print("No enhancement results available! Please run enhanced_upload_and_select() first.")
        return
    
    print("\nSAVING ALL RESULTS AS BMP FILES...")
    print("=" * 50)
    
    selection_map = {
        'original': 'Original',
        'enhanced_gabor': 'Fixed_Gabor',
        'clahe_adaptive': 'CLAHE_Adaptive', 
        'ridge_enhancement': 'Ridge_Enhancement',
        'contrast_stretch': 'Contrast_Stretch',
        'unsharp_mask': 'Unsharp_Mask',
        'multi_scale': 'Multi_Scale'
    }
    
    saved_files = []
    base_name = os.path.splitext(_current_enhancer.current_filename)[0]
    
    for method_key, method_name in selection_map.items():
        if method_key in _current_enhancer.current_results:
            try:
                image = _current_enhancer.current_results[method_key]
                bmp_filename = f"{base_name}_{method_name}.bmp"
                
                if image.dtype != np.uint8:
                    image = image.astype(np.uint8)
                
                success = cv2.imwrite(bmp_filename, image)
                if success:
                    saved_files.append(bmp_filename)
                    print(f"Saved: {bmp_filename}")
                else:
                    print(f"Failed to save: {bmp_filename}")
                    
            except Exception as e:
                print(f"Error saving {method_name}: {str(e)}")
    
    print(f"\nSAVED {len(saved_files)} BMP FILES!")
    print("Files saved in Colab storage:")
    for filename in saved_files:
        print(f"   • {filename}")
    
    print(f"\nTo download specific files, use:")
    print(f"   files.download('filename.bmp')")

def list_saved_files():
    bmp_files = [f for f in os.listdir('.') if f.endswith('.bmp')]
    
    if bmp_files:
        print(f"\nFOUND {len(bmp_files)} BMP FILES:")
        print("=" * 40)
        for i, filename in enumerate(bmp_files, 1):
            print(f"{i}. {filename}")
        print("\nTo download a file, use:")
        print("   files.download('filename.bmp')")
    else:
        print("No BMP files found in current directory")

def show_clean_result(number):
    global _current_enhancer
    
    if not hasattr(_current_enhancer, 'current_results') or not _current_enhancer.current_results:
        print("No enhancement results available! Please run enhanced_upload_and_select() first.")
        return

    if number not in [0, 1, 2, 3, 4, 5, 6]:
        print("Invalid selection! Please choose a number from 0 to 6.")
        return

    selection_map = {
        0: ('original', 'Original'),
        1: ('enhanced_gabor', 'Fixed_Gabor'),
        2: ('clahe_adaptive', 'CLAHE_Adaptive'),
        3: ('ridge_enhancement', 'Ridge_Enhancement'),
        4: ('contrast_stretch', 'Contrast_Stretch'),
        5: ('unsharp_mask', 'Unsharp_Mask'),
        6: ('multi_scale', 'Multi_Scale')
    }

    method_key, method_name = selection_map[number]

    try:
        if method_key == 'original':
            selected_image = _current_enhancer.current_results['original']
        else:
            selected_image = _current_enhancer.current_results[method_key]

        plt.figure(figsize=(10, 10))
        plt.imshow(selected_image, cmap='gray')
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.tight_layout(pad=0)
        plt.show()

        print(f"Showing clean result: {method_name}")

    except Exception as e:
        print(f"Error displaying result: {str(e)}")

_current_enhancer = EnhancedFingerprintSelector()

def enhanced_upload_and_select():
    global _current_enhancer
    
    print("\nUPLOAD YOUR FINGERPRINT - ENHANCED WITH SELECTION!")
    print("=" * 70)
    print("1. Upload your fingerprint image")
    print("2. View all numbered enhancement results") 
    print("3. Select your preferred result by number")
    print("4. Download as BMP file")
    print("=" * 70)

    uploaded = files.upload()

    if not uploaded:
        print("No files uploaded!")
        return

    for filename in uploaded.keys():
        print(f"\nProcessing: {filename}")
        print("-" * 50)

        try:
            image = cv2.imread(filename)
            if image is None:
                image = np.array(Image.open(filename))
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            original = image.copy()
            if len(original.shape) == 3:
                original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            else:
                original_gray = original

            print(f"Image size: {image.shape[1]}x{image.shape[0]} pixels")

            results = {}
            methods_to_test = list(_current_enhancer.methods.keys())

            print("\nApplying all enhancement methods...")
            for method in methods_to_test:
                enhanced = _current_enhancer.enhance_image(image, method)
                results[method] = enhanced

            _current_enhancer.current_results = results.copy()
            _current_enhancer.current_results['original'] = original_gray
            _current_enhancer.current_filename = filename

            _current_enhancer.display_numbered_results(original_gray, results, filename)

            print(f"\nMAKE YOUR SELECTION:")
            print("Type: select_and_save(NUMBER)")
            print("Example: select_and_save(1)  # for Fixed Gabor")
            print("Available numbers: 0, 1, 2, 3, 4, 5, 6")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

print("\nENHANCED FINGERPRINT TOOL WITH SELECTION READY!")
print("=" * 70)
print("Fixed blank output issue!")
print("Added numbering for easy selection")
print("Save selected result as BMP file")
print()
print("HOW TO USE:")
print("1  enhanced_upload_and_select()  - Upload and see numbered results")
print("2  select_and_save(NUMBER)       - Choose and download as BMP")
print("3  show_clean_result(NUMBER)     - Show ONLY clean enhanced image")
print("4  quick_clean_view()            - Quick upload & clean view")
print("5  save_all_results()            - Save all results as BMP files")
print("6  list_saved_files()            - Show saved BMP files")
print("7  download_file('filename.bmp') - Download specific file")
print()
print("Example Usage:")
print("   enhanced_upload_and_select()   # Upload and view options")
print("   show_clean_result(1)           # Show ONLY clean enhanced image")
print("   select_and_save(1)             # Select option 1 and save as BMP")
print("   quick_clean_view()             # Quick clean view")
print("=" * 70)