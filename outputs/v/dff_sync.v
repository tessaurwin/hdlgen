module DFF(
    input  logic clk,
    input  logic d,
    input  logic rst_n,
    output logic q
);

always @(posedge clk) begin
    if (rst_n == 0) begin
        q <= 1'b0;
    end else begin
        q <= d;
    end
end

endmodule
