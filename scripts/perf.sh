sudo perf script record -F997 --call-graph dwarf,16384 -e cpu-clock ./target/debug/nn reduced.csv age
sudo chmod +r perf.data
