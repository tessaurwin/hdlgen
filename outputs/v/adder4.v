module ripple_carry_adder(a, b, cin, sum, cout);
  input [3:0] a, b;
  input cin;
  output [3:0] sum;
  output cout;
  
  always @(*) begin
    sum = a + b + cin;
    cout = (a[3] & b[3]) | (a[2] & b[2]) | (a[1] & b[1]) | (a[0] & b[0]);
  end
endmodule
