import sys
import os

def setup_refer_lib():
    """
    Hàm này tự động tìm và thêm thư mục 'refer/pytorch-grad-cam' vào sys.path
    để các file khác có thể import thư viện bên ngoài dễ dàng.
    """
    # Lấy đường dẫn hiện tại (src/module_going)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Lùi về src/
    src_dir = os.path.dirname(current_dir)
    # Lùi về thư mục gốc (AIL303m_project)
    project_root = os.path.dirname(src_dir)
    
    # Trỏ tới thư viện trong refer
    gradcam_lib_path = os.path.join(project_root, 'refer', 'pytorch-grad-cam')
    
    # Add vào system path
    if gradcam_lib_path not in sys.path:
        sys.path.append(gradcam_lib_path)
        
    return project_root
