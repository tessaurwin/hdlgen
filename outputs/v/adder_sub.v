module adder_subtractor(a, b, sub, result, cout);
  input [7:0] a, b;
  input sub;
  output [7:0] result;
  output cout;

  always @* begin
    if (sub) begin
      result = a - b;
      cout = a[7] ^ b[7];
    end else begin
      result = a + b;
      cout = a[7] & b[7];
    end
  end
endmodule
