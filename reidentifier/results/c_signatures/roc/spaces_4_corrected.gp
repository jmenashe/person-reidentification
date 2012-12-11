set xrange [0:1]; set yrange [0:1];
set xlabel "False Positive Rate";
set ylabel "True Positive Rate";
set title "Color Signature ROC with Illumination Correction";
set datafile separator ',';
set key right center;
plot \
  "yuv_4_corrected.csv" u 1:2 title 'YUV' with lines, \
  "hsv_4_corrected.csv" u 1:2 title 'HSV' with lines, \
  "lab_4_corrected.csv" u 1:2 title 'LAB' with lines, \
  "rgb_4_corrected.csv" u 1:2 title 'RGB' with lines;
pause -1
