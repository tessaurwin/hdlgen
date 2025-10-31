module magnitude_comparator(a, b, lt, eq, gt);
  input [3:0] a;
  input [3:0] b;
  output reg lt;
  output reg eq;
  output reg gt;

  always @(*) begin
    if (a > b) begin
      lt = 1'b1;
      eq = 1'b0;
      gt = 1'b0;
    end else if (a < b) begin
      lt = 1'b0;
      eq = 1'b0;
      gt = 1'b1;
    end else begin
      lt = 1'b0;
      eq = 1'b1;
      gt = 1'b0;
    end
  end
endmodule
