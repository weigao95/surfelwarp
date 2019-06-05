//
// Created by wei on 3/30/18.
//

#pragma once

//Use materialized jtj?
#define USE_MATERIALIZED_JTJ

//Use dense solver maps
#define USE_DENSE_SOLVER_MAPS

//Use huber weights in sparse feature term to deal with outlier
#define USE_FEATURE_HUBER_WEIGHT

//If the node graph are too far-awary from each other, they are unlikely valid
#define CLIP_FARAWAY_NODEGRAPH_PAIR


//Clip the image term if they are too large
//#define USE_IMAGE_HUBER_WEIGHT
#define d_density_map_cutoff 0.3f
#define d_foreground_cutoff 0.1f

//Use rendered or input rgba map as solver input
#define USE_RENDERED_RGBA_MAP_SOLVER


//The value of invalid index map
#define d_invalid_index 0xFFFFFFFF

//The device accessed constant for find corresponded depth pairs
#define d_correspondence_normal_dot_threshold 0.7f
#define d_correspondence_distance_threshold 0.03f
#define d_correspondence_distance_threshold_square (d_correspondence_distance_threshold * d_correspondence_distance_threshold)

//The device accessed constant for valid color pixels
#define d_valid_color_dot_threshold 0.5

//The upperbound of alignment error
#define d_maximum_alignment_error 0.1f

//The weight between different terms, these are deprecated
#define lambda_smooth 2.0f
#define lambda_smooth_square (lambda_smooth * lambda_smooth)

#define lambda_density 0.0f
#define lambda_density_square (lambda_density * lambda_density)

#define lambda_foreground 0.0f
#define lambda_foreground_square (lambda_foreground * lambda_foreground)

#define lambda_feature 0.0f
#define lambda_feature_square (lambda_feature * lambda_feature)