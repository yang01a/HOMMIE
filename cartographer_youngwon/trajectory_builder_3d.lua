-- Copyright 2016 The Cartographer Authors
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

MAX_3D_RANGE = 3.5  -- TurtleBot3 Waffle LIDAR의 최대 범위에 맞게 설정
INTENSITY_THRESHOLD = 40

TRAJECTORY_BUILDER_3D = {
  min_range = 0.3,  -- 최소 범위를 LIDAR의 작동 범위에 맞게 설정
  max_range = MAX_3D_RANGE,
  num_accumulated_range_data = 1,
  voxel_filter_size = 0.1,  -- 필터 크기 조정

  high_resolution_adaptive_voxel_filter = {
    max_length = 0.5,  -- 고해상도 필터의 최대 길이
    min_num_points = 100,  -- 포인트 수 감소
    max_range = 2.0,  -- 고해상도 최대 범위
  },

  low_resolution_adaptive_voxel_filter = {
    max_length = 1.0,  -- 저해상도 필터의 최대 길이
    min_num_points = 150,  -- 포인트 수 감소
    max_range = MAX_3D_RANGE,
  },

  use_online_correlative_scan_matching = true,  -- 온라인 상관관계 스캔 매칭 사용
  real_time_correlative_scan_matcher = {
    linear_search_window = 0.1,  -- 선형 검색 창
    angular_search_window = math.rad(0.5),  -- 각도 검색 창 조정
    translation_delta_cost_weight = 1e-1,
    rotation_delta_cost_weight = 1e-1,
  },

  ceres_scan_matcher = {
    occupied_space_weight_0 = 1.,
    occupied_space_weight_1 = 6.,
    intensity_cost_function_options_0 = {
        weight = 0.5,
        huber_scale = 0.1,  -- Huber 스케일 조정
        intensity_threshold = INTENSITY_THRESHOLD,
    },
    translation_weight = 10.,  -- 이동 가중치 조정
    rotation_weight = 1e3,  -- 회전 가중치 조정
    only_optimize_yaw = false,
    ceres_solver_options = {
      use_nonmonotonic_steps = false,
      max_num_iterations = 10,  -- 최대 반복 횟수 감소
      num_threads = 1,
    },
  },

  motion_filter = {
    max_time_seconds = 0.3,  -- 최대 시간 조정
    max_distance_meters = 0.05,  -- 최대 거리 조정
    max_angle_radians = 0.003,  -- 최대 각도 조정
  },

  rotational_histogram_size = 120,

  imu_gravity_time_constant = 10.,
  pose_extrapolator = {
    use_imu_based = false,
    constant_velocity = {
      imu_gravity_time_constant = 10.,
      pose_queue_duration = 0.001,
    },
    imu_based = {
      pose_queue_duration = 5.,
      gravity_constant = 9.806,
      pose_translation_weight = 1.,
      pose_rotation_weight = 1.,
      imu_acceleration_weight = 1.,
      imu_rotation_weight = 1.,
      odometry_translation_weight = 1.,
      odometry_rotation_weight = 1.,
      solver_options = {
        use_nonmonotonic_steps = false;
        max_num_iterations = 10;
        num_threads = 1;
      },
    },
  },

  submaps = {
    high_resolution = 0.05,  -- 서브맵의 고해상도
    high_resolution_max_range = 10.,  -- 서브맵의 최대 고해상도 범위
    low_resolution = 0.15,  -- 서브맵의 저해상도
    num_range_data = 160,
    range_data_inserter = {
      hit_probability = 0.55,
      miss_probability = 0.49,
      num_free_space_voxels = 2,
      intensity_threshold = INTENSITY_THRESHOLD,
    },
  },

  use_intensities = true,  -- 강도 사용 설정
}

