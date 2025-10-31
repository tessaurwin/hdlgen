module synch_ram(
    input clk,
    input we,
    input [2:0] addr,
    input [7:0] din,
    output reg [7:0] dout
);

// Initialize the memory array with all zeros
reg [7:0] mem[8][8];

always @(posedge clk) begin
    if (we) begin
        // Write data to memory
        mem[addr][0] <= din;
    end
end

always @(*) begin
    dout = mem[addr][0];
end

endmodule
