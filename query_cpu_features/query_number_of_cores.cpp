#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <cstdio>
#include <cassert>
#include <vector>

bool get_windows_version()
{
    OSVERSIONINFOEXW osInfo = { 0 };
    osInfo.dwOSVersionInfoSize = sizeof(osInfo);

    typedef LONG(WINAPI* RtlGetVersionPtr)(PRTL_OSVERSIONINFOW);
    RtlGetVersionPtr RtlGetVersion = (RtlGetVersionPtr)GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlGetVersion");

    if (RtlGetVersion == nullptr) {
        perror("Failed to locate RtlGetVersion function.");
        return false;
    }
    if (RtlGetVersion(reinterpret_cast<PRTL_OSVERSIONINFOW>(&osInfo)) != 0) {
        perror("Failed to retrieve version information.");
        return false;
    }

    const bool is_server = (osInfo.wProductType == VER_NT_SERVER || osInfo.wProductType == VER_NT_DOMAIN_CONTROLLER);
    if (osInfo.dwMajorVersion == 10) {
        if (is_server) {
            if (osInfo.dwBuildNumber >= 20348) {
                puts("Windows Server 2022 detected.");
                return true;
            } else if (osInfo.dwBuildNumber >= 17763) {
                puts("Windows Server 2019 detected.");
            } else {
                puts("An older version of Windows Server detected (Windows Server 2016 or earlier).");
            }
        } else {
            if (osInfo.dwBuildNumber >= 22000) {
                puts("Windows 11 detected.");
                return true;
            }
            else {
                puts("Windows 10 detected.");
            }
        }
    }
    else {
        puts("An unsupported Windows version detected.");
    }

    printf("Version: %d.%d\n", osInfo.dwMajorVersion, osInfo.dwMinorVersion);
    printf("Build Number: %d\n", osInfo.dwBuildNumber);
    return false;
}

void get_processor_info(WORD target_group)
{
    DWORD buffer_size = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &buffer_size);
    if (buffer_size == 0) {
        perror("Failed to get buffer size for processor information.");
        return;
    }
    std::vector<BYTE> buffer(buffer_size);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buffer.data(), &buffer_size)) {
        perror("Failed to get processor information.");
        return;
    }

    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buffer.data();
    int p_cores = 0;
    int e_cores = 0;
    int logical_processor_count = 0;
    const bool query_all_processor_groups = (target_group == ALL_PROCESSOR_GROUPS);

    while (buffer_size > 0) {
        if (info->Relationship == RelationProcessorCore) {
            if (query_all_processor_groups || ((info->Processor.GroupCount > 0) && (info->Processor.GroupMask[0].Group == target_group))) {
                DWORD_PTR mask = info->Processor.GroupMask[0].Mask;
                int lp_count = 0;
                for (; mask; mask &= (mask - 1)) {
                    lp_count++;
                }
                if (lp_count) {
                    logical_processor_count += lp_count;
                    if (info->Processor.EfficiencyClass == 0) {
                        e_cores++;
                    } else {
                        p_cores++;
                    }
                }
            }
        }
        buffer_size -= info->Size;
        info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)((BYTE*)info + info->Size);
    }
    // on systems without the heterogenious cores EfficiencyClass always equals to zero
    if (!p_cores) {
        p_cores = e_cores;
        e_cores = 0;
    }

    printf("Physical p-cores in group #0x%x : %d\n", target_group, p_cores);
    printf("Physical e-cores in group  #0x%x : %d\n", target_group, e_cores);
    printf("Logical processors in group  #0x%x : %d\n", target_group, logical_processor_count);
}

int main()
{
    const bool windows11 = get_windows_version();
    DWORD current_group = 0;
    if (!windows11) {
        const DWORD active_group_count = GetActiveProcessorGroupCount();
        if (active_group_count > 1) {
            PROCESSOR_NUMBER proc_number;
            GetCurrentProcessorNumberEx(&proc_number);
            current_group = proc_number.Group;
#ifndef NDEBUG
            SYSTEM_INFO sys_info;
            GetSystemInfo(&sys_info);
            assert(sys_info.dwNumberOfProcessors == GetActiveProcessorCount(current_group));
#endif
        }
    } else {
        current_group = ALL_PROCESSOR_GROUPS;
    }
    get_processor_info((WORD)current_group);
    return 0;
}