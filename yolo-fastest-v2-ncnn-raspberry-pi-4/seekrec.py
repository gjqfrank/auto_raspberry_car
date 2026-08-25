import cv2
import numpy as np

# ================== 核心函数：找长方形、返回4个顶点 ==================
def find_rectangle_vertices(image):
    # 1. 转灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 2. 高斯模糊（降噪）
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 3. 边缘检测
    edges = cv2.Canny(blur, 50, 150)

    # 4. 找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rect_vertices = None  # 存储长方形4个顶点

    # 5. 遍历轮廓，找长方形
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:  # 过滤小噪点（可根据你的长方形大小调整）
            continue

        # 多边形逼近
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # 长方形 = 4个顶点 + 凸多边形
        if len(approx) == 4 and cv2.isContourConvex(approx):
            rect_vertices = approx
            break

    return rect_vertices


# ================== 主程序：打开摄像头实时识别 ==================
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # 0=默认摄像头

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 获取长方形4个顶点
        vertices = find_rectangle_vertices(frame)

        if vertices is not None:
            # 把顶点转成方便使用的格式
            pts = vertices.reshape(-1, 2)
            print("✅ 长方形4个顶点坐标：")
            for i, (x, y) in enumerate(pts):
                print(f"  顶点{i+1}: ({x}, {y})")
                cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)  # 画顶点

            # 画轮廓
            cv2.drawContours(frame, [vertices], -1, (0, 255, 0), 2)

        # 显示画面
        cv2.imshow("Rectangle Detector", frame)

        # 按 Q 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()