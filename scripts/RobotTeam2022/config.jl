ignorecols = [:latitude,
              :longitude,
              :ilat,
              :ilon,
              :unix_dt,
              :utc_dt,
              :category,
              :predye_postdye,
              :times,
              :row_index,
              ]

targets_vars = [:Br,
                :CDOM,
                :CO,
                :Ca,
                :Chl,
                :ChlRed,
                :Cl,
                :HDO,
                :HDO_percent,
                :NH4,
                :NO3,
                :Na,
                :OB,
                :RefFuel,
                :SSC,
                :Salinity3488,
                :Salinity3490,
                :SpCond,
                :TDS,
                :TRYP,
                :Temp3488,
                :Temp3489,
                :Temp3490,
                :Turb3488,
                :Turb3489,
                :Turb3490,
                :bg,
                :bgm,
                :pH,
                :pH_mV,
                ]


targetsDict = Dict(
    :Temp3488 => ["°C", "Temperature (3488)"],
    :pH => ["dimensionless", "pH", 0.0],
    :SpCond => ["μS/cm", "Conductivity"],
    :Turb3488 => ["FNU", "Turbidity (3488)"],
    :Br => ["mg/l", "Br⁻", 0.0],
    :Ca => ["mg/l", "Ca⁺⁺", 0.0],
    :Cl => ["mg/l", "Cl⁻", 0.0],
    :Na => ["mg/l", "Na⁺", 0.0],
    :NO3 => ["mg/l-N", "NO₃⁻", 0.0],
    :NH4 => ["mg/l-N", "NH₄⁺", 0.0],
    :HDO => ["mg/l", "Dissolved Oxygen", 0.0],
    :HDO_percent => ["% Sat", "Dissolved Oxygen", 0.0],
    :pH_mV => ["mV", "pH"],
#    :Salinity3488 => ["PSS", "Salinity (3488)"],
#    :TDS => ["mg/l", "Total Dissolved Solids", 0.0],
    :Temp3489 => ["°C", "Temperature (3489)"],
    :bg => ["ppb", "Blue-Green Algae fresh water (Phycocyanin)", 0.0],
    :bgm => ["ppb", "Blue-Green Algae salt water (Phycoerythrin)", 0.0],
    #:CDOM => ["ppb", "Colored dissolved organic matter (CDOM)", 0.0],
    :CDOM => ["ppb", "CDOM", 0.0],
    :Chl => ["μg/l", "Chlorophyll A", 0.0],
    :ChlRed => ["μg/l", "Chlorophyll A with Red Excitation", 0.0],
    :Turb3489 => ["FNU", "Turbidity (3489)"],
    :Temp3490 => ["°C", "Temperature (3490)"],
    :CO => ["ppb", "Crude Oil", 0.0],
    :OB => ["ppb", "Optical Brighteners", 0.0],
    :RefFuel => ["ppb", "Refined Fuels", 0.0],
    :TRYP => ["ppb", "Tryptophan", 0.0],
    :Turb3490 => ["FNU", "Turbidity (3490)"],
#    :Salinity3490 => ["PSS", "Salinity (3490)"],
#    :TDS => ["mg/l", "Suspended Sediment Concentration", 0.0],
)



wavelengths = [390.960, 392.270, 393.590, 394.910, 396.230, 397.550, 398.870, 400.180, 401.500, 402.820, 404.140, 405.460, 406.780, 408.100, 409.420, 410.740, 412.060, 413.380, 414.700, 416.020, 417.340, 418.660, 419.980, 421.300, 422.620, 423.940, 425.260, 426.580, 427.910, 429.230, 430.550, 431.870, 433.190, 434.510, 435.840, 437.160, 438.480, 439.800, 441.120, 442.450, 443.770, 445.090, 446.410, 447.740, 449.060, 450.380, 451.710, 453.030, 454.350, 455.680, 457.000, 458.320, 459.650, 460.970, 462.300, 463.620, 464.940, 466.270, 467.590, 468.920, 470.240, 471.570, 472.890, 474.220, 475.540, 476.870, 478.190, 479.520, 480.840, 482.170, 483.500, 484.820, 486.150, 487.480, 488.800, 490.130, 491.450, 492.780, 494.110, 495.430, 496.760, 498.090, 499.420, 500.740, 502.070, 503.400, 504.730, 506.050, 507.380, 508.710, 510.040, 511.370, 512.700, 514.020, 515.350, 516.680, 518.010, 519.340, 520.670, 522.000, 523.330, 524.660, 525.990, 527.310, 528.640, 529.970, 531.300, 532.630, 533.960, 535.300, 536.630, 537.960, 539.290, 540.620, 541.950, 543.280, 544.610, 545.940, 547.270, 548.600, 549.940, 551.270, 552.600, 553.930, 555.260, 556.600, 557.930, 559.260, 560.590, 561.920, 563.260, 564.590, 565.920, 567.260, 568.590, 569.920, 571.260, 572.590, 573.920, 575.260, 576.590, 577.920, 579.260, 580.590, 581.930, 583.260, 584.600, 585.930, 587.260, 588.600, 589.930, 591.270, 592.600, 593.940, 595.280, 596.610, 597.950, 599.280, 600.620, 601.950, 603.290, 604.630, 605.960, 607.300, 608.640, 609.970, 611.310, 612.650, 613.980, 615.320, 616.660, 617.990, 619.330, 620.670, 622.010, 623.340, 624.680, 626.020, 627.360, 628.700, 630.040, 631.370, 632.710, 634.050, 635.390, 636.730, 638.070, 639.410, 640.750, 642.090, 643.420, 644.760, 646.100, 647.440, 648.780, 650.120, 651.460, 652.800, 654.140, 655.480, 656.830, 658.170, 659.510, 660.850, 662.190, 663.530, 664.870, 666.210, 667.550, 668.900, 670.240, 671.580, 672.920, 674.260, 675.600, 676.950, 678.290, 679.630, 680.970, 682.320, 683.660, 685.000, 686.350, 687.690, 689.030, 690.380, 691.720, 693.060, 694.410, 695.750, 697.090, 698.440, 699.780, 701.130, 702.470, 703.820, 705.160, 706.510, 707.850, 709.200, 710.540, 711.890, 713.230, 714.580, 715.920, 717.270, 718.610, 719.960, 721.310, 722.650, 724.000, 725.340, 726.690, 728.040, 729.380, 730.730, 732.080, 733.420, 734.770, 736.120, 737.470, 738.810, 740.160, 741.510, 742.860, 744.200, 745.550, 746.900, 748.250, 749.600, 750.950, 752.290, 753.640, 754.990, 756.340, 757.690, 759.040, 760.390, 761.740, 763.090, 764.440, 765.790, 767.140, 768.490, 769.840, 771.190, 772.540, 773.890, 775.240, 776.590, 777.940, 779.290, 780.640, 781.990, 783.340, 784.690, 786.050, 787.400, 788.750, 790.100, 791.450, 792.800, 794.160, 795.510, 796.860, 798.210, 799.570, 800.920, 802.270, 803.620, 804.980, 806.330, 807.680, 809.040, 810.390, 811.740, 813.100, 814.450, 815.800, 817.160, 818.510, 819.870, 821.220, 822.580, 823.930, 825.280, 826.640, 827.990, 829.350, 830.700, 832.060, 833.420, 834.770, 836.130, 837.480, 838.840, 840.190, 841.550, 842.910, 844.260, 845.620, 846.980, 848.330, 849.690, 851.050, 852.400, 853.760, 855.120, 856.470, 857.830, 859.190, 860.550, 861.900, 863.260, 864.620, 865.980, 867.340, 868.690, 870.050, 871.410, 872.770, 874.130, 875.490, 876.850, 878.210, 879.560, 880.920, 882.280, 883.640, 885.000, 886.360, 887.720, 889.080, 890.440, 891.800, 893.160, 894.520, 895.880, 897.240, 898.600, 899.970, 901.330, 902.690, 904.050, 905.410, 906.770, 908.130, 909.490, 910.860, 912.220, 913.580, 914.940, 916.300, 917.670, 919.030, 920.390, 921.750, 923.120, 924.480, 925.840, 927.210, 928.570, 929.930, 931.300, 932.660, 934.020, 935.390, 936.750, 938.120, 939.480, 940.840, 942.210, 943.570, 944.940, 946.300, 947.670, 949.030, 950.400, 951.760, 953.130, 954.490, 955.860, 957.220, 958.590, 959.960, 961.320, 962.690, 964.050, 965.420, 966.790, 968.150, 969.520, 970.890, 972.250, 973.620, 974.990, 976.350, 977.720, 979.090, 980.460, 981.820, 983.190, 984.560, 985.930, 987.300, 988.660, 990.030, 991.400, 992.770, 994.140, 995.510, 996.880, 998.250, 999.610, 1000.980, 1002.350, 1003.720, 1005.090, 1006.460, 1007.830, 1009.200, 1010.570]


downwelling_wavelengths = [339.708, 340.091, 340.474, 340.856, 341.239, 341.622, 342.005, 342.387, 342.770, 343.153, 343.535, 343.918, 344.300, 344.683, 345.065, 345.448, 345.830, 346.212, 346.594, 346.977, 347.359, 347.741, 348.123, 348.505, 348.887, 349.269, 349.651, 350.033, 350.414, 350.796, 351.178, 351.560, 351.941, 352.323, 352.704, 353.086, 353.467, 353.849, 354.230, 354.611, 354.993, 355.374, 355.755, 356.136, 356.517, 356.898, 357.279, 357.660, 358.041, 358.422, 358.803, 359.184, 359.565, 359.945, 360.326, 360.707, 361.087, 361.468, 361.848, 362.229, 362.609, 362.990, 363.370, 363.750, 364.130, 364.511, 364.891, 365.271, 365.651, 366.031, 366.411, 366.791, 367.171, 367.551, 367.931, 368.310, 368.690, 369.070, 369.449, 369.829, 370.208, 370.588, 370.967, 371.347, 371.726, 372.106, 372.485, 372.864, 373.243, 373.622, 374.002, 374.381, 374.760, 375.139, 375.518, 375.896, 376.275, 376.654, 377.033, 377.412, 377.790, 378.169, 378.547, 378.926, 379.305, 379.683, 380.061, 380.440, 380.818, 381.196, 381.575, 381.953, 382.331, 382.709, 383.087, 383.465, 383.843, 384.221, 384.599, 384.977, 385.355, 385.732, 386.110, 386.488, 386.865, 387.243, 387.621, 387.998, 388.375, 388.753, 389.130, 389.508, 389.885, 390.262, 390.639, 391.016, 391.394, 391.771, 392.148, 392.525, 392.901, 393.278, 393.655, 394.032, 394.409, 394.785, 395.162, 395.539, 395.915, 396.292, 396.668, 397.045, 397.421, 397.798, 398.174, 398.550, 398.926, 399.302, 399.679, 400.055, 400.431, 400.807, 401.183, 401.559, 401.934, 402.310, 402.686, 403.062, 403.438, 403.813, 404.189, 404.564, 404.940, 405.315, 405.691, 406.066, 406.442, 406.817, 407.192, 407.567, 407.942, 408.318, 408.693, 409.068, 409.443, 409.818, 410.192, 410.567, 410.942, 411.317, 411.692, 412.066, 412.441, 412.816, 413.190, 413.565, 413.939, 414.313, 414.688, 415.062, 415.436, 415.811, 416.185, 416.559, 416.933, 417.307, 417.681, 418.055, 418.429, 418.803, 419.177, 419.551, 419.924, 420.298, 420.672, 421.045, 421.419, 421.792, 422.166, 422.539, 422.913, 423.286, 423.659, 424.033, 424.406, 424.779, 425.152, 425.525, 425.898, 426.271, 426.644, 427.017, 427.390, 427.763, 428.136, 428.508, 428.881, 429.254, 429.626, 429.999, 430.371, 430.744, 431.116, 431.489, 431.861, 432.233, 432.605, 432.978, 433.350, 433.722, 434.094, 434.466, 434.838, 435.210, 435.582, 435.953, 436.325, 436.697, 437.069, 437.440, 437.812, 438.184, 438.555, 438.927, 439.298, 439.669, 440.041, 440.412, 440.783, 441.154, 441.526, 441.897, 442.268, 442.639, 443.010, 443.381, 443.752, 444.123, 444.493, 444.864, 445.235, 445.606, 445.976, 446.347, 446.717, 447.088, 447.458, 447.829, 448.199, 448.569, 448.940, 449.310, 449.680, 450.050, 450.420, 450.790, 451.160, 451.530, 451.900, 452.270, 452.640, 453.010, 453.379, 453.749, 454.119, 454.488, 454.858, 455.227, 455.597, 455.966, 456.336, 456.705, 457.074, 457.443, 457.813, 458.182, 458.551, 458.920, 459.289, 459.658, 460.027, 460.396, 460.764, 461.133, 461.502, 461.871, 462.239, 462.608, 462.976, 463.345, 463.713, 464.082, 464.450, 464.819, 465.187, 465.555, 465.923, 466.291, 466.660, 467.028, 467.396, 467.764, 468.132, 468.499, 468.867, 469.235, 469.603, 469.971, 470.338, 470.706, 471.073, 471.441, 471.808, 472.176, 472.543, 472.911, 473.278, 473.645, 474.012, 474.379, 474.747, 475.114, 475.481, 475.848, 476.215, 476.581, 476.948, 477.315, 477.682, 478.049, 478.415, 478.782, 479.148, 479.515, 479.881, 480.248, 480.614, 480.981, 481.347, 481.713, 482.079, 482.446, 482.812, 483.178, 483.544, 483.910, 484.276, 484.642, 485.007, 485.373, 485.739, 486.105, 486.470, 486.836, 487.201, 487.567, 487.932, 488.298, 488.663, 489.029, 489.394, 489.759, 490.124, 490.490, 490.855, 491.220, 491.585, 491.950, 492.315, 492.680, 493.044, 493.409, 493.774, 494.139, 494.503, 494.868, 495.232, 495.597, 495.961, 496.326, 496.690, 497.055, 497.419, 497.783, 498.147, 498.512, 498.876, 499.240, 499.604, 499.968, 500.332, 500.696, 501.059, 501.423, 501.787, 502.151, 502.514, 502.878, 503.241, 503.605, 503.968, 504.332, 504.695, 505.059, 505.422, 505.785, 506.148, 506.511, 506.875, 507.238, 507.601, 507.964, 508.326, 508.689, 509.052, 509.415, 509.778, 510.140, 510.503, 510.866, 511.228, 511.591, 511.953, 512.316, 512.678, 513.040, 513.403, 513.765, 514.127, 514.489, 514.851, 515.213, 515.575, 515.937, 516.299, 516.661, 517.023, 517.385, 517.746, 518.108, 518.470, 518.831, 519.193, 519.554, 519.916, 520.277, 520.638, 521.000, 521.361, 521.722, 522.083, 522.445, 522.806, 523.167, 523.528, 523.889, 524.249, 524.610, 524.971, 525.332, 525.693, 526.053, 526.414, 526.774, 527.135, 527.495, 527.856, 528.216, 528.577, 528.937, 529.297, 529.657, 530.017, 530.378, 530.738, 531.098, 531.458, 531.818, 532.177, 532.537, 532.897, 533.257, 533.617, 533.976, 534.336, 534.695, 535.055, 535.414, 535.774, 536.133, 536.492, 536.852, 537.211, 537.570, 537.929, 538.288, 538.647, 539.006, 539.365, 539.724, 540.083, 540.442, 540.801, 541.159, 541.518, 541.877, 542.235, 542.594, 542.952, 543.311, 543.669, 544.027, 544.386, 544.744, 545.102, 545.460, 545.818, 546.177, 546.535, 546.893, 547.250, 547.608, 547.966, 548.324, 548.682, 549.039, 549.397, 549.755, 550.112, 550.470, 550.827, 551.185, 551.542, 551.899, 552.257, 552.614, 552.971, 553.328, 553.685, 554.042, 554.399, 554.756, 555.113, 555.470, 555.827, 556.184, 556.540, 556.897, 557.254, 557.610, 557.967, 558.323, 558.680, 559.036, 559.392, 559.749, 560.105, 560.461, 560.817, 561.173, 561.529, 561.885, 562.241, 562.597, 562.953, 563.309, 563.665, 564.020, 564.376, 564.732, 565.087, 565.443, 565.798, 566.154, 566.509, 566.865, 567.220, 567.575, 567.930, 568.286, 568.641, 568.996, 569.351, 569.706, 570.061, 570.416, 570.770, 571.125, 571.480, 571.835, 572.189, 572.544, 572.899, 573.253, 573.607, 573.962, 574.316, 574.671, 575.025, 575.379, 575.733, 576.087, 576.442, 576.796, 577.150, 577.504, 577.857, 578.211, 578.565, 578.919, 579.273, 579.626, 579.980, 580.333, 580.687, 581.040, 581.394, 581.747, 582.101, 582.454, 582.807, 583.160, 583.513, 583.866, 584.220, 584.573, 584.926, 585.278, 585.631, 585.984, 586.337, 586.690, 587.042, 587.395, 587.747, 588.100, 588.452, 588.805, 589.157, 589.510, 589.862, 590.214, 590.566, 590.919, 591.271, 591.623, 591.975, 592.327, 592.679, 593.030, 593.382, 593.734, 594.086, 594.437, 594.789, 595.141, 595.492, 595.844, 596.195, 596.547, 596.898, 597.249, 597.600, 597.952, 598.303, 598.654, 599.005, 599.356, 599.707, 600.058, 600.409, 600.760, 601.110, 601.461, 601.812, 602.162, 602.513, 602.864, 603.214, 603.564, 603.915, 604.265, 604.616, 604.966, 605.316, 605.666, 606.016, 606.366, 606.716, 607.066, 607.416, 607.766, 608.116, 608.466, 608.815, 609.165, 609.515, 609.864, 610.214, 610.563, 610.913, 611.262, 611.612, 611.961, 612.310, 612.659, 613.008, 613.358, 613.707, 614.056, 614.405, 614.754, 615.102, 615.451, 615.800, 616.149, 616.497, 616.846, 617.195, 617.543, 617.892, 618.240, 618.588, 618.937, 619.285, 619.633, 619.982, 620.330, 620.678, 621.026, 621.374, 621.722, 622.070, 622.418, 622.765, 623.113, 623.461, 623.809, 624.156, 624.504, 624.851, 625.199, 625.546, 625.894, 626.241, 626.588, 626.936, 627.283, 627.630, 627.977, 628.324, 628.671, 629.018, 629.365, 629.712, 630.059, 630.405, 630.752, 631.099, 631.445, 631.792, 632.138, 632.485, 632.831, 633.178, 633.524, 633.870, 634.217, 634.563, 634.909, 635.255, 635.601, 635.947, 636.293, 636.639, 636.985, 637.330, 637.676, 638.022, 638.368, 638.713, 639.059, 639.404, 639.750, 640.095, 640.440, 640.786, 641.131, 641.476, 641.821, 642.167, 642.512, 642.857, 643.202, 643.547, 643.891, 644.236, 644.581, 644.926, 645.271, 645.615, 645.960, 646.304, 646.649, 646.993, 647.338, 647.682, 648.026, 648.371, 648.715, 649.059, 649.403, 649.747, 650.091, 650.435, 650.779, 651.123, 651.467, 651.810, 652.154, 652.498, 652.841, 653.185, 653.529, 653.872, 654.215, 654.559, 654.902, 655.245, 655.589, 655.932, 656.275, 656.618, 656.961, 657.304, 657.647, 657.990, 658.333, 658.676, 659.018, 659.361, 659.704, 660.046, 660.389, 660.731, 661.074, 661.416, 661.759, 662.101, 662.443, 662.785, 663.128, 663.470, 663.812, 664.154, 664.496, 664.838, 665.180, 665.521, 665.863, 666.205, 666.547, 666.888, 667.230, 667.571, 667.913, 668.254, 668.596, 668.937, 669.278, 669.620, 669.961, 670.302, 670.643, 670.984, 671.325, 671.666, 672.007, 672.348, 672.689, 673.029, 673.370, 673.711, 674.051, 674.392, 674.732, 675.073, 675.413, 675.754, 676.094, 676.434, 676.774, 677.114, 677.455, 677.795, 678.135, 678.475, 678.815, 679.154, 679.494, 679.834, 680.174, 680.513, 680.853, 681.193, 681.532, 681.872, 682.211, 682.551, 682.890, 683.229, 683.568, 683.908, 684.247, 684.586, 684.925, 685.264, 685.603, 685.942, 686.280, 686.619, 686.958, 687.297, 687.635, 687.974, 688.313, 688.651, 688.989, 689.328, 689.666, 690.005, 690.343, 690.681, 691.019, 691.357, 691.695, 692.033, 692.371, 692.709, 693.047, 693.385, 693.723, 694.060, 694.398, 694.736, 695.073, 695.411, 695.748, 696.086, 696.423, 696.760, 697.098, 697.435, 697.772, 698.109, 698.446, 698.783, 699.120, 699.457, 699.794, 700.131, 700.467, 700.804, 701.141, 701.478, 701.814, 702.151, 702.487, 702.824, 703.160, 703.496, 703.833, 704.169, 704.505, 704.841, 705.177, 705.513, 705.849, 706.185, 706.521, 706.857, 707.193, 707.528, 707.864, 708.200, 708.535, 708.871, 709.206, 709.542, 709.877, 710.212, 710.548, 710.883, 711.218, 711.553, 711.888, 712.223, 712.558, 712.893, 713.228, 713.563, 713.898, 714.233, 714.567, 714.902, 715.237, 715.571, 715.906, 716.240, 716.575, 716.909, 717.243, 717.577, 717.912, 718.246, 718.580, 718.914, 719.248, 719.582, 719.916, 720.250, 720.583, 720.917, 721.251, 721.585, 721.918, 722.252, 722.585, 722.919, 723.252, 723.585, 723.919, 724.252, 724.585, 724.918, 725.251, 725.585, 725.918, 726.250, 726.583, 726.916, 727.249, 727.582, 727.915, 728.247, 728.580, 728.912, 729.245, 729.577, 729.910, 730.242, 730.574, 730.907, 731.239, 731.571, 731.903, 732.235, 732.567, 732.899, 733.231, 733.563, 733.895, 734.226, 734.558, 734.890, 735.221, 735.553, 735.884, 736.216, 736.547, 736.879, 737.210, 737.541, 737.872, 738.204, 738.535, 738.866, 739.197, 739.528, 739.859, 740.190, 740.520, 740.851, 741.182, 741.512, 741.843, 742.174, 742.504, 742.835, 743.165, 743.495, 743.826, 744.156, 744.486, 744.816, 745.146, 745.476, 745.806, 746.136, 746.466, 746.796, 747.126, 747.456, 747.785, 748.115, 748.445, 748.774, 749.104, 749.433, 749.763, 750.092, 750.421, 750.750, 751.080, 751.409, 751.738, 752.067, 752.396, 752.725, 753.054, 753.383, 753.711, 754.040, 754.369, 754.697, 755.026, 755.355, 755.683, 756.012, 756.340, 756.668, 756.997, 757.325, 757.653, 757.981, 758.309, 758.637, 758.965, 759.293, 759.621, 759.949, 760.277, 760.604, 760.932, 761.260, 761.587, 761.915, 762.242, 762.570, 762.897, 763.225, 763.552, 763.879, 764.206, 764.533, 764.860, 765.187, 765.514, 765.841, 766.168, 766.495, 766.822, 767.149, 767.475, 767.802, 768.128, 768.455, 768.781, 769.108, 769.434, 769.761, 770.087, 770.413, 770.739, 771.065, 771.391, 771.717, 772.043, 772.369, 772.695, 773.021, 773.347, 773.672, 773.998, 774.324, 774.649, 774.975, 775.300, 775.626, 775.951, 776.276, 776.602, 776.927, 777.252, 777.577, 777.902, 778.227, 778.552, 778.877, 779.202, 779.527, 779.851, 780.176, 780.501, 780.825, 781.150, 781.474, 781.799, 782.123, 782.447, 782.772, 783.096, 783.420, 783.744, 784.068, 784.392, 784.716, 785.040, 785.364, 785.688, 786.012, 786.336, 786.659, 786.983, 787.306, 787.630, 787.953, 788.277, 788.600, 788.923, 789.247, 789.570, 789.893, 790.216, 790.539, 790.862, 791.185, 791.508, 791.831, 792.154, 792.477, 792.799, 793.122, 793.445, 793.767, 794.090, 794.412, 794.734, 795.057, 795.379, 795.701, 796.024, 796.346, 796.668, 796.990, 797.312, 797.634, 797.956, 798.277, 798.599, 798.921, 799.243, 799.564, 799.886, 800.207, 800.529, 800.850, 801.172, 801.493, 801.814, 802.135, 802.457, 802.778, 803.099, 803.420, 803.741, 804.062, 804.383, 804.703, 805.024, 805.345, 805.665, 805.986, 806.307, 806.627, 806.948, 807.268, 807.588, 807.909, 808.229, 808.549, 808.869, 809.189, 809.509, 809.829, 810.149, 810.469, 810.789, 811.109, 811.428, 811.748, 812.068, 812.387, 812.707, 813.026, 813.346, 813.665, 813.984, 814.304, 814.623, 814.942, 815.261, 815.580, 815.899, 816.218, 816.537, 816.856, 817.175, 817.494, 817.812, 818.131, 818.449, 818.768, 819.086, 819.405, 819.723, 820.042, 820.360, 820.678, 820.996, 821.314, 821.633, 821.951, 822.269, 822.586, 822.904, 823.222, 823.540, 823.858, 824.175, 824.493, 824.810, 825.128, 825.445, 825.763, 826.080, 826.397, 826.715, 827.032, 827.349, 827.666, 827.983, 828.300, 828.617, 828.934, 829.251, 829.568, 829.884, 830.201, 830.518, 830.834, 831.151, 831.467, 831.784, 832.100, 832.416, 832.733, 833.049, 833.365, 833.681, 833.997, 834.313, 834.629, 834.945, 835.261, 835.577, 835.892, 836.208, 836.524, 836.839, 837.155, 837.470, 837.786, 838.101, 838.416, 838.732, 839.047, 839.362, 839.677, 839.992, 840.307, 840.622, 840.937, 841.252, 841.567, 841.882, 842.196, 842.511, 842.825, 843.140, 843.455, 843.769, 844.083, 844.398, 844.712, 845.026, 845.340, 845.655, 845.969, 846.283, 846.597, 846.911, 847.224, 847.538, 847.852, 848.166, 848.479, 848.793, 849.106, 849.420, 849.733, 850.047, 850.360, 850.673, 850.987, 851.300, 851.613, 851.926, 852.239, 852.552, 852.865, 853.178, 853.491, 853.803, 854.116, 854.429, 854.741, 855.054, 855.367, 855.679, 855.991, 856.304, 856.616, 856.928, 857.240, 857.553, 857.865, 858.177, 858.489, 858.801, 859.113, 859.424, 859.736, 860.048, 860.360, 860.671, 860.983, 861.294, 861.606, 861.917, 862.228, 862.540, 862.851, 863.162, 863.473, 863.784, 864.095, 864.406, 864.717, 865.028, 865.339, 865.650, 865.961, 866.271, 866.582, 866.892, 867.203, 867.513, 867.824, 868.134, 868.444, 868.755, 869.065, 869.375, 869.685, 869.995, 870.305, 870.615, 870.925, 871.235, 871.545, 871.854, 872.164, 872.474, 872.783, 873.093, 873.402, 873.711, 874.021, 874.330, 874.639, 874.949, 875.258, 875.567, 875.876, 876.185, 876.494, 876.803, 877.111, 877.420, 877.729, 878.038, 878.346, 878.655, 878.963, 879.272, 879.580, 879.888, 880.197, 880.505, 880.813, 881.121, 881.429, 881.737, 882.045, 882.353, 882.661, 882.969, 883.277, 883.584, 883.892, 884.200, 884.507, 884.815, 885.122, 885.430, 885.737, 886.044, 886.352, 886.659, 886.966, 887.273, 887.580, 887.887, 888.194, 888.501, 888.807, 889.114, 889.421, 889.728, 890.034, 890.341, 890.647, 890.954, 891.260, 891.566, 891.873, 892.179, 892.485, 892.791, 893.097, 893.403, 893.709, 894.015, 894.321, 894.627, 894.933, 895.238, 895.544, 895.849, 896.155, 896.460, 896.766, 897.071, 897.377, 897.682, 897.987, 898.292, 898.597, 898.902, 899.207, 899.512, 899.817, 900.122, 900.427, 900.732, 901.036, 901.341, 901.646, 901.950, 902.255, 902.559, 902.863, 903.168, 903.472, 903.776, 904.080, 904.384, 904.688, 904.992, 905.296, 905.600, 905.904, 906.208, 906.511, 906.815, 907.119, 907.422, 907.726, 908.029, 908.333, 908.636, 908.939, 909.243, 909.546, 909.849, 910.152, 910.455, 910.758, 911.061, 911.364, 911.667, 911.969, 912.272, 912.575, 912.877, 913.180, 913.482, 913.785, 914.087, 914.389, 914.692, 914.994, 915.296, 915.598, 915.900, 916.202, 916.504, 916.806, 917.108, 917.410, 917.712, 918.013, 918.315, 918.616, 918.918, 919.219, 919.521, 919.822, 920.124, 920.425, 920.726, 921.027, 921.328, 921.629, 921.930, 922.231, 922.532, 922.833, 923.134, 923.434, 923.735, 924.036, 924.336, 924.637, 924.937, 925.238, 925.538, 925.838, 926.138, 926.439, 926.739, 927.039, 927.339, 927.639, 927.939, 928.239, 928.538, 928.838, 929.138, 929.437, 929.737, 930.037, 930.336, 930.635, 930.935, 931.234, 931.533, 931.833, 932.132, 932.431, 932.730, 933.029, 933.328, 933.627, 933.926, 934.224, 934.523, 934.822, 935.120, 935.419, 935.717, 936.016, 936.314, 936.613, 936.911, 937.209, 937.507, 937.806, 938.104, 938.402, 938.700, 938.997, 939.295, 939.593, 939.891, 940.189, 940.486, 940.784, 941.081, 941.379, 941.676, 941.974, 942.271, 942.568, 942.865, 943.163, 943.460, 943.757, 944.054, 944.351, 944.648, 944.944, 945.241, 945.538, 945.835, 946.131, 946.428, 946.724, 947.021, 947.317, 947.613, 947.910, 948.206, 948.502, 948.798, 949.094, 949.390, 949.686, 949.982, 950.278, 950.574, 950.870, 951.165, 951.461, 951.756, 952.052, 952.347, 952.643, 952.938, 953.233, 953.529, 953.824, 954.119, 954.414, 954.709, 955.004, 955.299, 955.594, 955.889, 956.184, 956.478, 956.773, 957.068, 957.362, 957.657, 957.951, 958.245, 958.540, 958.834, 959.128, 959.422, 959.716, 960.010, 960.304, 960.598, 960.892, 961.186, 961.480, 961.774, 962.067, 962.361, 962.655, 962.948, 963.241, 963.535, 963.828, 964.121, 964.415, 964.708, 965.001, 965.294, 965.587, 965.880, 966.173, 966.466, 966.759, 967.051, 967.344, 967.637, 967.929, 968.222, 968.514, 968.807, 969.099, 969.391, 969.684, 969.976, 970.268, 970.560, 970.852, 971.144, 971.436, 971.728, 972.020, 972.311, 972.603, 972.895, 973.186, 973.478, 973.769, 974.061, 974.352, 974.643, 974.935, 975.226, 975.517, 975.808, 976.099, 976.390, 976.681, 976.972, 977.263, 977.554, 977.844, 978.135, 978.426, 978.716, 979.007, 979.297, 979.587, 979.878, 980.168, 980.458, 980.748, 981.038, 981.328, 981.618, 981.908, 982.198, 982.488, 982.778, 983.068, 983.357, 983.647, 983.936, 984.226, 984.515, 984.805, 985.094, 985.383, 985.673, 985.962, 986.251, 986.540, 986.829, 987.118, 987.407, 987.696, 987.984, 988.273, 988.562, 988.850, 989.139, 989.428, 989.716, 990.004, 990.293, 990.581, 990.869, 991.157, 991.446, 991.734, 992.022, 992.310, 992.597, 992.885, 993.173, 993.461, 993.749, 994.036, 994.324, 994.611, 994.899, 995.186, 995.473, 995.761, 996.048, 996.335, 996.622, 996.909, 997.196, 997.483, 997.770, 998.057, 998.344, 998.631, 998.917, 999.204, 999.490, 999.777, 1000.063, 1000.350, 1000.636, 1000.922, 1001.209, 1001.495, 1001.781, 1002.067, 1002.353, 1002.639, 1002.925, 1003.211, 1003.497, 1003.782, 1004.068, 1004.354, 1004.639, 1004.925, 1005.210, 1005.496, 1005.781, 1006.066, 1006.352, 1006.637, 1006.922, 1007.207, 1007.492, 1007.777, 1008.062, 1008.347, 1008.631, 1008.916, 1009.201, 1009.485, 1009.770, 1010.055, 1010.339, 1010.623, 1010.908, 1011.192, 1011.476, 1011.760, 1012.045, 1012.329, 1012.613, 1012.897, 1013.181, 1013.464, 1013.748, 1014.032, 1014.316, 1014.599, 1014.883, 1015.166, 1015.450, 1015.733, 1016.016, 1016.300, 1016.583, 1016.866, 1017.149, 1017.432, 1017.715, 1017.998, 1018.281, 1018.564, 1018.847, 1019.130, 1019.412, 1019.695, 1019.977, 1020.260, 1020.542, 1020.825, 1021.107, 1021.389, 1021.672, 1021.954, 1022.236, 1022.518, 1022.800, 1023.082, 1023.364, 1023.646, 1023.927, 1024.209, 1024.491]


name_replacements = Dict()
for i ∈ 1:size(wavelengths, 1)
    name_replacements["λ_$(i)_rad"] = "Radiance at $(wavelengths[i]) nm"
end

for i ∈ 1:size(downwelling_wavelengths, 1)
    name_replacements["λ_downwelling_$(i)"] = "Downwelling Irradiance at $(downwelling_wavelengths[i]) nm"
end


# add derived metrics
name_replacements["mNDWI"] = "Modified Normalized Difference Water Index"
name_replacements["rad_mNDWI"] = "Modified Normalized Difference Water Index (radiance)"

name_replacements["NDVI"] = "Normalized Difference Vegetative Index"
name_replacements["rad_NDVI"] = "Normalized Difference Vegetative Index (radiance)"

name_replacements["SR "] = "Simple Ratio"
name_replacements["rad_SR"] = "Simple Ratio (radiance)"

name_replacements["EVI"] = "Enhanced Vegetative Index"
name_replacements["rad_EVI"] = "Enhanced Vegetative Index (radiance)"

name_replacements["AVRI"] = "Atmospherical Resistant Vegetative Index"
name_replacements["rad_AVRI"] = "Atmospherical Resistant Vegetative Index (radiance)"

name_replacements["NDVI_705"] = "Red Edge Normalized Difference Vegetation Index"
name_replacements["rad_NDVI_705"] = "Red Edge Normalized Difference Vegetation Index (radiance)"

name_replacements["MSR_705"] = "Modified Red Edge Simple Ratio"
name_replacements["rad_MSR_705"] = "Modified Red Edge Simple Ratio (radiance)"

name_replacements["MNDVI"] = "Modified Red Edge Normalized Difference Vegetation Index"
name_replacements["rad_MNDVI"] = "Modified Red Edge Normalized Difference Vegetation Index"

name_replacements["VOG1"] = "Vogelmann Red Edge Index"
name_replacements["rad_VOG1"] = "Vogelmann Red Edge Index (radiance)"

name_replacements["VOG2"] = "Vogelmann Red Edge Index 2"
name_replacements["rad_VOG2"] = "Vogelmann Red Edge Index 2 (radiance)"

name_replacements["VOG3"] = "Vogelmann Red Edge Index 3"
name_replacements["rad_VOG3"] = "Vogelmann Red Edge Index 3 (radiance)"

name_replacements["PRI"] = "Photochemical Reflectance Index"
name_replacements["rad_PRI"] = "Photochemical Reflectance Index (radiance)"

name_replacements["SIPI"] = "Structure Intensive Pigment Index"
name_replacements["rad_SIPI"] = "Structure Intensive Pigment Index (radiance)"

name_replacements["PSRI"] = "Plant Senescence Reflectance Index"
name_replacements["rad_PSRI"] = "Plant Senescence Reflectance Index (radiance)"

name_replacements["CRT1"] = "Carotenoid Reflectance Index"
name_replacements["rad_CRT1"] = "Carotenoid Reflectance Index (radiance)"

name_replacements["CRT2"] = "Carotenoid Reflectance Index 2"
name_replacements["rad_CRT2"] = "Carotenoid Reflectance Index 2 (radiance)"

name_replacements["ARI1"] = "Anthocyanin Reflectance Index"
name_replacements["rad_ARI1"] = "Anthocyanin Reflectance Index (radiance)"

name_replacements["ARI2"] = "Anthocyanin Reflectance Index 2"
name_replacements["rad_ARI2"] = "Anthocyanin Reflectance Index 2 (radiance)"

name_replacements["WBI"] = "Water Band Index"
name_replacements["rad_WBI"] = "Water Band Index (radiance)"

name_replacements["MCRI"] = "Modified Chlorophyll Absorption Reflectance Index"
name_replacements["rad_MCRI"] = "Modified Chlorophyll Absorption Reflectance Index (radiance)"

name_replacements["TCARI"] = "Transformed Chlorophyll Absorption Reflectance Index"
name_replacements["rad_TCARI"] = "Transformed Chlorophyll Absorption Reflectance Index (radiance)"




