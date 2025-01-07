include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",                        -- 맵 프레임
  tracking_frame = "base_link",             -- 추적 프레임 (보통 "base_link")
  published_frame = "base_link",            -- 게시된 프레임 (보통 "base_link")
  odom_frame = "odom",                      -- 오돔 프레임
  provide_odom_frame = true,                -- 오돔 프레임 제공 활성화
  publish_frame_projected_to_2d = false,    -- 2D로 투사하여 게시 비활성화
  use_pose_extrapolator = true,             -- 포즈 외삽기 사용 활성화
  use_odometry = false,                     -- 오도메트리 사용 비활성화
  use_nav_sat = false,                      -- 내비게이션 위성 사용 비활성화
  use_landmarks = false,                    -- 랜드마크 사용 비활성화
  num_laser_scans = 1,                      -- LIDAR 사용 설정
  num_multi_echo_laser_scans = 0,           -- 멀티 에코 LIDAR 스캔 사용 비활성화
  num_subdivisions_per_laser_scan = 1,      -- LIDAR 스캔당 세분화 수 (10에서 1로 변경)
  num_point_clouds = 0,                     -- 포인트 클라우드 사용 비활성화
  lookup_transform_timeout_sec = 0.2,       -- 변환 조회 타임아웃
  submap_publish_period_sec = 0.3,          -- 서브맵 게시 주기
  pose_publish_period_sec = 5e-3,           -- 포즈 게시 주기
  trajectory_publish_period_sec = 30e-3,    -- 경로 게시 주기
  rangefinder_sampling_ratio = 1.0,         -- 거리 측정기 샘플링 비율
  odometry_sampling_ratio = 1.0,            -- 오도메트리 샘플링 비율
  fixed_frame_pose_sampling_ratio = 1.0,    -- 고정 프레임 포즈 샘플링 비율
  imu_sampling_ratio = 0.0,                 -- IMU 샘플링 비율 비활성화
  landmarks_sampling_ratio = 1.0,           -- 랜드마크 샘플링 비율
}

MAP_BUILDER.use_trajectory_builder_2d = true -- 2D 경로 빌더 사용

-- 누적 범위 데이터를 낮추어 실시간 데이터 처리 활성화
TRAJECTORY_BUILDER_2D.num_accumulated_range_data = 1

-- 서브맵 생성에 필요한 레이저 스캔 데이터 개수를 낮춥니다.
TRAJECTORY_BUILDER_2D.submaps.num_range_data = 45

-- LIDAR의 최대 및 최소 범위를 설정합니다.
TRAJECTORY_BUILDER_2D.max_range = 10.0  -- 최대 감지 범위 (10미터)
TRAJECTORY_BUILDER_2D.min_range = 0.1   -- 최소 감지 범위 (10cm)

-- IMU 데이터를 비활성화합니다.
TRAJECTORY_BUILDER_2D.use_imu_data = false

-- Motion filter를 추가하여 데이터 처리 효율을 개선합니다.
TRAJECTORY_BUILDER_2D.motion_filter = {
  max_distance_meters = 0.15,  -- 15cm 이동 시 새로운 데이터 사용
  max_angle_radians = 0.1,    -- 0.1 라디안 회전 시 새로운 데이터 사용
  max_time_seconds = 0.5,     -- 0.5초마다 새로운 데이터 사용
}

return options
