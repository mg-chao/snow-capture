#[inline(always)]
pub(crate) fn swap_bgra_to_rgba(pixel: u32) -> u32 {
    ((pixel & 0x0000_00FF) << 16)
        | (pixel & 0x0000_FF00)
        | ((pixel & 0x00FF_0000) >> 16)
        | (pixel & 0xFF00_0000)
}

pub(crate) unsafe fn convert_bgra_to_rgba_scalar_unchecked(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
) {
    let mut src_px = src as *const u32;
    let mut dst_px = dst as *mut u32;
    let mut remaining = pixel_count;

    while remaining >= 8 {
        unsafe {
            let p0 = std::ptr::read_unaligned(src_px);
            let p1 = std::ptr::read_unaligned(src_px.add(1));
            let p2 = std::ptr::read_unaligned(src_px.add(2));
            let p3 = std::ptr::read_unaligned(src_px.add(3));
            let p4 = std::ptr::read_unaligned(src_px.add(4));
            let p5 = std::ptr::read_unaligned(src_px.add(5));
            let p6 = std::ptr::read_unaligned(src_px.add(6));
            let p7 = std::ptr::read_unaligned(src_px.add(7));

            std::ptr::write_unaligned(dst_px, swap_bgra_to_rgba(p0));
            std::ptr::write_unaligned(dst_px.add(1), swap_bgra_to_rgba(p1));
            std::ptr::write_unaligned(dst_px.add(2), swap_bgra_to_rgba(p2));
            std::ptr::write_unaligned(dst_px.add(3), swap_bgra_to_rgba(p3));
            std::ptr::write_unaligned(dst_px.add(4), swap_bgra_to_rgba(p4));
            std::ptr::write_unaligned(dst_px.add(5), swap_bgra_to_rgba(p5));
            std::ptr::write_unaligned(dst_px.add(6), swap_bgra_to_rgba(p6));
            std::ptr::write_unaligned(dst_px.add(7), swap_bgra_to_rgba(p7));
        }

        src_px = unsafe { src_px.add(8) };
        dst_px = unsafe { dst_px.add(8) };
        remaining -= 8;
    }

    while remaining != 0 {
        unsafe {
            let pixel = std::ptr::read_unaligned(src_px);
            std::ptr::write_unaligned(dst_px, swap_bgra_to_rgba(pixel));
        }

        src_px = unsafe { src_px.add(1) };
        dst_px = unsafe { dst_px.add(1) };
        remaining -= 1;
    }
}
