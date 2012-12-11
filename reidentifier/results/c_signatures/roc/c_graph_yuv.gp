set xrange [0:1]; set yrange [0:1];
set xlabel "False Positive Rate";
set ylabel "True Positive Rate";
set title "Color Signature ROC Curves";
set datafile separator ',';
set key right center;
plot \
  "c_signature_yuv_1.csv" u 1:2 title '1 Slice YUV' with lines, \
  "c_signature_yuv_2.csv" u 1:2 title '2 Slices YUV' with lines, \
  "c_signature_yuv_3.csv" u 1:2 title '3 Slices YUV' with lines, \
  "c_signature_yuv_4.csv" u 1:2 title '4 Slices YUV' with lines, \
  "c_signature_yuv_5.csv" u 1:2 title '5 Slices YUV' with lines;
pause -1
