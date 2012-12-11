set xrange [0:1]; set yrange [0:1];
set xlabel "True Positive Rate";
set ylabel "False Positive Rate";
set title "Color Signature ROC Curves";
set datafile separator ',';
set key right center;
plot \
  "c_signature_1.csv" u 1:2 title '1 Slice' with lines, \
  "c_signature_2.csv" u 1:2 title '2 Slices' with lines, \
  "c_signature_3.csv" u 1:2 title '3 Slices' with lines, \
  "c_signature_4.csv" u 1:2 title '4 Slices' with lines, \
  "c_signature_5.csv" u 1:2 title '5 Slices' with lines;
pause -1
