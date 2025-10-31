module mux2(a, b, sel, y);
    input a, b;
    input sel;
    output y;

    assign y = (sel == 1) ? a : b;
endmodule
