estimated_time_delay[v, time + 1] = real_time_delay[v, time + 1] + np.random.normal(0, sigma_time_delay[
    v, time], 1)
estimated_doppler_frequency[v, time + 1] = real_doppler_frequency[v, time + 1] + np.random.normal(0,
                                                                                                  estimated_distance[
                                                                                                      v, time + 1] = 0.5 * config_parameter.c *
                                                                                                                     estimated_time_delay[
                                                                                                                         v, time + 1]  # precoding matrix for this time is for the estimation_next time
estimated_velocity_between_norm = (estimated_distance[v, time + 1] - estimated_distance[
    v, time]) / config_parameter.Radar_measure_slot
sigma_doppler[
                                                                                                      v, time],
real_time_delay[vehicle, i] = 2 * real_distance_list[v, time] / config_parameter.                                                                                                  1)