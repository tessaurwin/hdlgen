module up_counter(
    input logic clk,
    input logic rst_n,
    input logic en,
    output logic [7:0] q
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        q <= 8'b0;
    end else if (en) begin
        q <= q + 1;
    end
end

endmodule
