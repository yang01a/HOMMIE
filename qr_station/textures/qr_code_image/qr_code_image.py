import qrcode

# QR 코드 데이터 설정 (여기서는 예시로 좌표값을 넣습니다)
data = "1.0,2.0,0.0"  # x, y, z 좌표

# QR 코드 생성
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data(data)
qr.make(fit=True)

# 이미지를 생성하여 저장
img = qr.make_image(fill='black', back_color='white')
img.save("/home/opulent/catkin_ws/src/qr_code.png")
