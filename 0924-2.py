import cv2
import numpy as np
import math
# 骰子拍照定位系統
# 目前功能：
#
# 1. 開啟攝影機
# 2. 即時預覽畫面
# 3. 按下「1」拍攝一張照片
# 4. 分析這一張照片
# 5. 偵測紅色 / 藍色骰子
# 6. 找出骰子中心
# 7. 以「畫面中心」作為 (0, 0)
# 8. 計算骰子 X / Y Pixel Offset
# 9. 計算骰子旋轉角度
# 10. 將結果以中文顯示

# 1. Camera 設定
# Camera 編號
CAMERA_ID = 1
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# 2. 骰子尺寸
DICE_REAL_SIZE_CM = 4.0

# 3. 畫面中心設定
USE_IMAGE_CENTER = True
MANUAL_CENTER_X = 640.0
MANUAL_CENTER_Y = 360.0

# 4. 顏色偵測參數
# 藍色 HSV 範圍
LOWER_BLUE = np.array([ 90,100,70])
UPPER_BLUE = np.array([130,255,255])

# 紅色 HSV 範圍
LOWER_RED_1 = np.array([ 0,100,70])
UPPER_RED_1 = np.array([10,255,255])
LOWER_RED_2 = np.array([170,100,70])
UPPER_RED_2 = np.array([180,255,255])


# 5. 骰子偵測參數
# 最小輪廓面積
MIN_AREA = 500
# 最大輪廓面積
MAX_AREA = 100000
# Bounding Box 長寬比
MIN_ASPECT_RATIO = 0.6
MAX_ASPECT_RATIO = 1.4
# 旋轉矩形長寬比
MIN_ROTATED_ASPECT_RATIO = 0.6


# 6. 顯示設定
# 拍照後是否顯示分析結果
SHOW_RESULT = True


# 7. 建立顏色 Mask
def create_color_masks(frame):
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # 藍色
    blue_mask = cv2.inRange(hsv,LOWER_BLUE,UPPER_BLUE)

    # 紅色
    red_mask_1 = cv2.inRange( hsv,LOWER_RED_1,UPPER_RED_1)
    red_mask_2 = cv2.inRange(hsv,LOWER_RED_2,UPPER_RED_2)
    red_mask = cv2.bitwise_or(red_mask_1,red_mask_2)

    # 去除小雜點、填補小洞
    kernel = np.ones((5, 5),np.uint8)

    blue_mask = cv2.morphologyEx(blue_mask,cv2.MORPH_OPEN,kernel)
    blue_mask = cv2.morphologyEx(blue_mask,cv2.MORPH_CLOSE,kernel)

    red_mask = cv2.morphologyEx(red_mask,cv2.MORPH_OPEN,kernel)
    red_mask = cv2.morphologyEx(red_mask,cv2.MORPH_CLOSE,kernel)

    return blue_mask, red_mask


# 8. 找骰子
def find_dice(mask, color_name):
    contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for contour in contours:
        # 面積
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue
        if area > MAX_AREA:
            continue

        # Bounding Rect
        x, y, w, h = cv2.boundingRect(contour)
        if h <= 0:
            continue

        aspect_ratio = w / float(h)

        # 正方形判斷
        if not (MIN_ASPECT_RATIO<= aspect_ratio<= MAX_ASPECT_RATIO):
            continue

        # 旋轉矩形
        rect = cv2.minAreaRect(contour)

        (rect_center,rect_size,rect_angle) = rect

        rect_cx, rect_cy = rect_center
        rect_w, rect_h = rect_size

        if rect_w <= 0 or rect_h <= 0:
            continue

        # 旋轉矩形長寬比

        min_side = min(rect_w,rect_h)
        max_side = max(rect_w,rect_h)

        rect_aspect_ratio = ( min_side / max_side)

        if (rect_aspect_ratio< MIN_ROTATED_ASPECT_RATIO):
            continue

        # Candidate

        dice = {
            "color": color_name,
            "center": (rect_cx,rect_cy),
            "width": rect_w,
            "height": rect_h,
            "area": area,
            "bbox": (x,y,w,h),
            "angle": rect_angle
        }

        candidates.append(dice)

    # 沒有候選
    if len(candidates) == 0:
        return None

    # 選擇面積最大的候選物件
    candidates.sort(key=lambda d: d["area"],reverse=True)
    return candidates[0]


# 9. 找出畫面中心
def get_image_center(frame):
    height, width = frame.shape[:2]

    if USE_IMAGE_CENTER:
        center_x = width / 2.0
        center_y = height / 2.0

    else:
        center_x = MANUAL_CENTER_X
        center_y = MANUAL_CENTER_Y

    return (center_x,center_y)


# 10. 計算 Pixel Offset
def calculate_pixel_offset(dice_x,dice_y,center_x,center_y):
    offset_x = (dice_x - center_x)
    offset_y = (dice_y - center_y)
    return (offset_x,offset_y)

# 11. 計算骰子旋轉角度
def calculate_dice_rotation(dice):
    angle = dice["angle"]
    width = dice["width"]
    height = dice["height"]
    if width < height:
        angle = angle + 90.0

    while angle > 90:
        angle -= 180

    while angle < -90:
        angle += 180

    return angle


# 12. 畫面中心
def draw_center_point(frame,center_x,center_y):
    cv2.circle(frame,(int(center_x),int(center_y)),6,(255, 255, 255),-1)
    cv2.putText(frame,"畫面中心 (0, 0)",(int(center_x + 10),int(center_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255, 255, 255),2)


# 13. 顯示中文資訊
def draw_detection_result(frame,dice,center_x,center_y,offset_x,offset_y,rotation_angle):
    color = dice["color"]
    center_dice_x, center_dice_y = (dice["center"])
    width = dice["width"]
    height = dice["height"]
    x, y, w, h = dice["bbox"]

    # 顏色
    if color == "BLUE":
        draw_color = (255,0,0)
    else:
        draw_color = (0,0,255)

    # Bounding Box
    cv2.rectangle(frame,(x, y),(x + w,y + h),draw_color,2)

 
    # 骰子中心
    cv2.circle(frame,(int(center_dice_x),int(center_dice_y)),7,(0, 255, 0),-1)

    # 畫面中心 → 骰子中心
    cv2.line(frame,(int(center_x),int(center_y)),(int(center_dice_x),int(center_dice_y)),(0, 255, 255),)

    # 顯示骰子名稱
    cv2.putText(frame,f"{color} DICE",(x,max(y - 10, 20)),cv2.FONT_HERSHEY_SIMPLEX,0.7,draw_color,2)

    # 資訊
    text_y = 30
    line_height = 30

    information = [
        f"Dice Center: "
        f"({center_dice_x:.1f}, "
        f"{center_dice_y:.1f}) px",

        f"X Offset: "
        f"{offset_x:+.1f} px",

        f"Y Offset: "
        f"{offset_y:+.1f} px",

        f"Rotation Angle: "
        f"{rotation_angle:+.2f} deg",

        f"Dice Size: "
        f"{width:.1f} x "
        f"{height:.1f} px",

        f"Real Dice Size: "
        f"{DICE_REAL_SIZE_CM:.1f} cm"]

    for text in information:

        cv2.putText(frame,text,(20,text_y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 255),2)
        text_y += line_height


# 14. 分析照片
def analyze_image(frame):
    # 取得畫面中心
    center_x, center_y = (get_image_center(frame))
    # 建立 Mask
    blue_mask, red_mask = (create_color_masks(frame))

    # 偵測藍色骰子
    blue_dice = find_dice(blue_mask,"BLUE")

    # 偵測紅色骰子
    red_dice = find_dice(red_mask,"RED")
    dice = None
    if blue_dice is not None:
        dice = blue_dice

    if red_dice is not None:
        if dice is None:
            dice = red_dice

        else:
            if (red_dice["area"]> dice["area"]):
                dice = red_dice

    # 沒找到
    if dice is None:
        draw_center_point(frame,center_x,center_y)
        cv2.putText(frame,"沒有偵測到骰子",(20, 40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2)

        return (None,center_x,center_y,None,None,None)

    # 骰子中心
    dice_x, dice_y = dice["center"]
    # Pixel Offset
    offset_x, offset_y = (calculate_pixel_offset(dice_x,dice_y,center_x,center_y))

    # 旋轉角度
    rotation_angle = (calculate_dice_rotation(dice))

    # 畫面中心
    draw_center_point(frame,center_x,center_y)

    # 畫分析結果
    draw_detection_result(frame,dice, center_x,center_y,offset_x,offset_y,rotation_angle)

    # 終端機輸出
    print()
    print("=" * 60)
    print("骰子分析結果")
    print("=" * 60)
    print(f"骰子顏色：{dice['color']}")
    print(f"畫面中心："f"({center_x:.1f}, {center_y:.1f}) px")
    print(f"骰子中心："f"({dice_x:.1f}, {dice_y:.1f}) px")
    print(f"X 偏移："f"{offset_x:+.1f} px")
    print(f"Y 偏移："f"{offset_y:+.1f} px")
    print(f"旋轉角度："f"{rotation_angle:+.2f}°")
    print(f"骰子影像尺寸："f"{dice['width']:.1f} × "f"{dice['height']:.1f} px")

    print(
        f"骰子實際尺寸："
        f"{DICE_REAL_SIZE_CM:.1f} × "
        f"{DICE_REAL_SIZE_CM:.1f} × "
        f"{DICE_REAL_SIZE_CM:.1f} cm"
    )

    print("實際距離換算：尚未標定")
    print("=" * 60)

    return (dice,center_x,center_y,offset_x,offset_y,rotation_angle)

# 15. 主程式
def main():
    # 開啟 Camera
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print("錯誤：無法開啟攝影機")
        return

    # 設定解析度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT)

    print()
    print("=" * 60)
    print("骰子拍照定位系統")
    print("=" * 60)

    print("攝影機解析度：",FRAME_WIDTH,"×",FRAME_HEIGHT)

    print(
        f"骰子實際尺寸："
        f"{DICE_REAL_SIZE_CM:.1f} cm × "
        f"{DICE_REAL_SIZE_CM:.1f} cm × "
        f"{DICE_REAL_SIZE_CM:.1f} cm"
    )

    print()
    print("操作方式：")
    print("  按 1 → 拍攝一張照片並分析")
    print("  按 Q → 離開程式")
    print()
    print("目前只分析 Pixel Offset 與旋轉角度")
    print("目前不進行實際距離回推")
    print("=" * 60)

    while True:
        # 取得目前影像
        ret, frame = cap.read()

        if not ret:
            print("錯誤：無法取得攝影機影像")
            break

        # 取得畫面中心
        center_x, center_y = (get_image_center(frame))

        # 顯示畫面中心
        draw_center_point(frame,center_x,center_y)

        # 顯示操作提示
        cv2.putText(frame,"按 1 拍照分析 / 按 Q 離開",(20, FRAME_HEIGHT - 20),
            cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 255, 255),2)

        # 顯示即時畫面
        cv2.imshow("Dice Camera",frame)

        # 等待鍵盤
        key = (cv2.waitKey(1)& 0xFF)

        # 按 1
        if key == ord("1"):
            print()
            print("正在拍攝照片...")

            ret, captured_frame = (cap.read())

            if not ret:
                print("錯誤：拍照失敗")
                continue

            # 分析這一張照片
            result = analyze_image(captured_frame)

            # 顯示分析結果
            if SHOW_RESULT:
                cv2.imshow("照片分析結果",captured_frame)
                print()
                print("分析結果已顯示。")
                print("按任意鍵返回即時畫面。")

                cv2.waitKey(0)
                cv2.destroyWindow("照片分析結果")

        # 按 Q
        elif key == ord("q"):
            break

    # 關閉
    cap.release()
    cv2.destroyAllWindows()



# 程式入口
if __name__ == "__main__":
    main()
