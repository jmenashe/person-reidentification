set terminal pdf
set output "colorspaces_4slice.pdf"
set yrange [0:632]
set xrange [0:632]
set xlabel "Minimum Rank";
set ylabel "Matches";
set title "Color Signature Ranks";
set datafile separator ',';
set key right center;
plot \
  "yuv_4.csv" u 1:2 title 'YUV' with lines, \
  "hsv_4.csv" u 1:2 title 'HSV' with lines, \
  "hsl_4.csv" u 1:2 title 'HSL' with lines, \
  "lab_4.csv" u 1:2 title 'Lab' with lines, \
  "rgb_4.csv" u 1:2 title 'RGB' with lines;
pause -1
