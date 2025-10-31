module and_gate(y,a,b);
  input a;
  input b;
  output y;
always @* begin
    y = a && b;
  end
endmodule