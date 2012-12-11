set terminal pdf;
set output "lab_rates.pdf"
set xrange [0:10]; set yrange [0:1];
set xlabel "Library Size";
set ylabel "Correct Identification Rate";
set title "Identification Rates on Lab Images";
set datafile separator ',';
set key right top;
plot \
  "success_rates_lab.csv" u 1:2 title 'Color Only' with lines, \
  "success_rates_lab_weighted.csv" u 1:2 title 'Color + Attributes' with lines;
pause -1
