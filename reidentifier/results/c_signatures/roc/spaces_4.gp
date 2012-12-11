set terminal pdf;
set output "colorspaces_4slice_roc.pdf"
set xrange [0:1]; set yrange [0:1];
set xlabel "False Positive Rate";
set ylabel "True Positive Rate";
set title "Color Signature ROC Curves";
set datafile separator ',';
set key right center;
plot \
  "yuv_4.csv" u 1:2 title 'YUV' with lines, \
  "hsv_4.csv" u 1:2 title 'HSV' with lines, \
  "hsl_4.csv" u 1:2 title 'HSL' with lines, \
  "lab_4.csv" u 1:2 title 'LAB' with lines, \
  "rgb_4.csv" u 1:2 title 'RGB' with lines;
pause -1
