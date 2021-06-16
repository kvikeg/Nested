python nested.py 

copy /Y trained_model.h5 ..\Poly
cd ..\Poly
python read_weights.py %1 --debug > printed_network.txt
copy /Y printed_network.txt ..\Nested
cd ..\Nested
python central_poly.py printed_network.txt
