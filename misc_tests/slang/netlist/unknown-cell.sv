module top (
    input  logic a,
    output logic y
);
    mdIVX8 u_buf (
        .A(a),
        .Y(y)
    );
endmodule