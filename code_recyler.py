
'''
 数据分块
'''
'''
dT_dt_chunks = torch.chunk(dT_dt, num_chunks, dim=0)
        dT_dxyt_chunks = torch.chunk(dT_dxyt, num_chunks, dim=0)
        dT_dxx_chunks = torch.chunk(dT_dxx, num_chunks, dim=0)
        dT_dyy_chunks = torch.chunk(dT_dyy, num_chunks, dim=0)
        print(f"6 Memory before backward: {torch.cuda.memory_allocated(self.device)} bytes")
        loss_equs = [self.pde_loss(u_inside,dT_dt,dT_dxyt, dT_dxx, dT_dyy) for \
                u_inside,dT_dt,dT_dxyt, dT_dxx, dT_dyy in zip(u_inside_chunks,dT_dt_chunks,dT_dxyt_chunks,dT_dxx_chunks,dT_dyy_chunks)]
        loss_equation = sum(loss_equs)
        # 最终loss
        loss = loss_equation + loss_boundary
        loss.backward()
'''