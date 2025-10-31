module serial_to_parallel(
    input clk,
    input rst_n,
    input serial_in,
    output reg [7:0] q
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        q <= 8'b0;
    end else begin
        q[0] <= serial_in;
        for (int i = 1; i < 8; i++) begin
            q[i] <= q[i-1];
        end
    end
end

endmodule
