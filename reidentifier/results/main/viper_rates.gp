set terminal pdf;
set output "viper_rates.pdf"
set xrange [0:30]; set yrange [0:1];
set xlabel "Library Size";
set ylabel "Correct Identification Rate";
set title "Identification Rates on VIPeR Images";
set datafile separator ',';
set key right top;
plot \
  "success_rates_viper.csv" u 1:2 title 'Color Only' with lines, \
  "success_rates_viper_weighted.csv" u 1:2 title 'Color + Attributes' with lines;
pause -1
